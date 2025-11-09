#!/usr/bin/env python3
"""
graph_ingest.py

Ingests documents under `ci_data_lake/<company>/` and extracts
entity-relationship triples using Gemini (LangChain Google GenAI),
then pushes them into Neo4j AuraDB.

Features:
- Gemini model fallback (tries 2.0 flash, then gemini-pro)
- Retry/backoff for LLM calls
- Basic PDF/DOCX reading (if python-docx / PyPDF2 installed)
- Marks files as ingested in Neo4j to avoid duplicates
- Console progress + final summary
"""

import os
import time
import math
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

# text splitter import (langchain updated package)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional libs for pdf/docx reading (best-effort)
try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

# -------------------------
# Configuration
# -------------------------
load_dotenv()
DATA_PATH = "ci_data_lake"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise SystemExit("Missing Neo4j credentials in .env (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD)")

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -------------------------
# Initialize LLM with fallback
# -------------------------
print("Initializing Gemini model (with fallback)...")
llm = None
llm_try_order = ["models/gemini-2.0-flash", "gemini-pro", "models/gemini-1.5-flash-8b"]

for model_name in llm_try_order:
    try:
        print(f"  -> Trying LLM model: {model_name}")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        # quick test call (no prompt) not always available; rely on runtime errors later
        break
    except Exception as e:
        print(f"  ⚠️ Model {model_name} failed to initialize: {e}")
        llm = None
        continue

if llm is None:
    raise SystemExit("Failed to initialize any Gemini model. Check your Google API key and available models.")

# -------------------------
# LLM Prompt
# -------------------------
ENTITY_PROMPT = """
Extract key competitive intelligence entities and relationships from the text below.
Return ONLY lines of triples in the exact form:
Entity1 -[relationship]-> Entity2

Examples:
Zoho CRM -[offers]-> AI Assistant
HubSpot -[charges]-> $50/month
Tech Nova -[competes_with]-> HubSpot

If there are also simple attributes (e.g., "pricing: $20/mo"), emit as:
Zoho CRM -[pricing]-> $20/mo

TEXT:
{chunk}
"""

# -------------------------
# Utilities: read file content (txt, md, pdf, docx)
# -------------------------
def read_text_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".docx" and docx:
        try:
            doc = docx.Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception:
            return ""
    if ext == ".pdf" and PyPDF2:
        try:
            text = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        except Exception:
            return ""
    # fallback: attempt plain read (binary->str)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

# -------------------------
# LLM wrapper with retry/backoff
# -------------------------
def llm_extract_triples(text, max_retries=5):
    prompt = ENTITY_PROMPT.format(chunk=text[:3000])  # limit chunk size
    backoff = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(prompt)
            # LangChain Google wrapper sometimes returns object with .content
            raw = getattr(response, "content", None) or getattr(response, "text", None) or str(response)
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            triples = []
            for ln in lines:
                # accept lines like: A -[rel]-> B
                if "-[" in ln and "]->" in ln:
                    try:
                        left = ln.split("-[")[0].strip().strip("()")
                        rel = ln.split("-[")[1].split("]->")[0].strip()
                        right = ln.split("]->")[1].strip().strip("()")
                        if left and rel and right:
                            triples.append((left, rel, right))
                    except Exception:
                        continue
            return triples
        except Exception as e:
            print(f"  Retry {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                print("  -> Giving up on this chunk.")
                return []
            sleep = backoff * (2 ** (attempt - 1))
            time.sleep(sleep)
    return []

# -------------------------
# Neo4j helper functions
# -------------------------
def mark_file_ingested(tx, company, filename):
    tx.run("""
        MERGE (f:IngestedFile {company:$company, filename:$filename})
        SET f.ingested_at = datetime()
    """, company=company, filename=filename)

def is_file_ingested(tx, company, filename):
    rec = tx.run("""
        MATCH (f:IngestedFile {company:$company, filename:$filename})
        RETURN f LIMIT 1
    """, company=company, filename=filename)
    return rec.peek() is not None

def push_triples(tx, company, triples):
    # Triples: list of (a, rel, b)
    for a, rel, b in triples:
        # MERGE company node
        tx.run("MERGE (c:Company {name:$company})", company=company)
        # MERGE entities
        tx.run("MERGE (e1:Entity {name:$a})", a=a)
        tx.run("MERGE (e2:Entity {name:$b})", b=b)
        # Company mentions entity
        tx.run("""
            MATCH (c:Company {name:$company}), (e1:Entity {name:$a})
            MERGE (c)-[:MENTIONS]->(e1)
        """, company=company, a=a)
        # Relationship between entities (store relationship type as relationship type if safe)
        rel_safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in rel).upper()
        # create relationship with a property for original rel
        tx.run(f"""
            MATCH (e1:Entity {{name:$a}}), (e2:Entity {{name:$b}})
            MERGE (e1)-[r:{rel_safe}]->(e2)
            SET r.type = $rel
        """, a=a, b=b, rel=rel)

# -------------------------
# Main ingestion pipeline
# -------------------------
def main():
    print("🚀 Starting Graph Ingestion (Neo4j)")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Data path not found: {DATA_PATH}")
        return

    stats = {"files_total": 0, "files_skipped": 0, "files_processed": 0, "triples_added": 0}
    with driver.session() as session:
        for company_dir in sorted(os.listdir(DATA_PATH)):
            company_path = os.path.join(DATA_PATH, company_dir)
            if not os.path.isdir(company_path):
                continue
            print(f"\n📂 Processing company folder: {company_dir}")
            for filename in sorted(os.listdir(company_path)):
                if not filename.lower().endswith((".txt", ".md", ".pdf", ".docx")):
                    print(f"  - Skipping (unsupported type): {filename}")
                    continue

                stats["files_total"] += 1
                file_path = os.path.join(company_path, filename)

                # Check already ingested
                already = session.read_transaction(is_file_ingested, company_dir, filename)
                if already:
                    print(f"  - Already ingested: {filename} (skipping)")
                    stats["files_skipped"] += 1
                    continue

                text = read_text_file(file_path)
                if not text or len(text.strip()) < 10:
                    print(f"  - Empty or unreadable file: {filename} (skipping)")
                    stats["files_skipped"] += 1
                    continue

                # Split into chunks and extract triples per chunk
                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                chunks = splitter.split_text(text)
                total_triples_for_file = 0
                for i, chunk in enumerate(chunks, start=1):
                    triples = llm_extract_triples(chunk)
                    if triples:
                        # push to neo4j in a write transaction
                        session.write_transaction(push_triples, company_dir, triples)
                        total_triples_for_file += len(triples)
                        stats["triples_added"] += len(triples)
                        print(f"    + chunk {i}: added {len(triples)} triples")
                    else:
                        print(f"    - chunk {i}: no triples extracted")

                # mark file as ingested if anything processed (or even if zero, to avoid reprocessing)
                session.write_transaction(mark_file_ingested, company_dir, filename)
                stats["files_processed"] += 1
                print(f"  ✅ Finished {filename}. Triples added from file: {total_triples_for_file}")

    # Summary
    print("\n🎉 Graph Ingestion Summary:")
    print(f"  Files found:     {stats['files_total']}")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Files skipped:   {stats['files_skipped']}")
    print(f"  Triples added:   {stats['triples_added']}")
    print("✅ Done. Check Neo4j AuraDB dashboard or run your Streamlit app to visualize graph.")

if __name__ == "__main__":
    main()
