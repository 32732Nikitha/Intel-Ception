import os
import shutil
from dotenv import load_dotenv
from neo4j import GraphDatabase
import spacy
import re

from langchain_community.document_loaders import DirectoryLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# --- 1. Configuration ---
load_dotenv()

DATA_PATH = "ci_data_lake"
DB_PATH = "db"

# Neo4j config - set in .env
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Initialize NLP
nlp = spacy.load("en_core_web_sm")

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# -------------------------
# Graph helper functions
# -------------------------
def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9 _\-.]", "", name).strip()


def create_company_if_not_exists(tx, company_name: str):
    tx.run(
        "MERGE (c:Company {name: $name}) RETURN c", name=company_name
    )


def create_entity_if_not_exists(tx, entity_name: str, entity_type: str):
    tx.run(
        "MERGE (e:Entity {name:$name}) "
        "SET e.type = $etype RETURN e",
        name=entity_name,
        etype=entity_type,
    )


def create_mention_relationship(tx, company_name: str, entity_name: str):
    tx.run(
        "MATCH (c:Company {name:$company}), (e:Entity {name:$entity}) "
        "MERGE (c)-[:MENTIONS]->(e)",
        company=company_name,
        entity=entity_name,
    )


def create_simple_relation(tx, subj: str, rel: str, obj: str):
    tx.run(
        "MERGE (a:Entity {name:$a}) MERGE (b:Entity {name:$b}) "
        "MERGE (a)-[r:%s]->(b) " % rel,
        a=subj,
        b=obj
    )


def extract_entities_from_text(text: str):
    doc = nlp(text)
    ents = []
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "MONEY", "PERCENT", "GPE", "DATE"):
            ents.append((ent.text.strip(), ent.label_))
    return ents


def infer_relationships_from_text(text: str, entities: list):
    # Very simple heuristics to create relations:
    # - If MONEY appears near an ORG or PRODUCT, create PRICE relation
    # - If percent appears near PRODUCT/ORG create PERFORMANCE relation
    relations = []
    lowered = text.lower()
    # find money tokens
    money_matches = []
    for ent_text, ent_type in entities:
        if ent_type == "MONEY":
            money_matches.append(ent_text)

    # naive window-based relationship inference
    for money in money_matches:
        # find sentences with money, and check other entities in that sentence
        for sent in [s.text for s in nlp(text).sents]:
            if money in sent:
                for ent_text, ent_type in entities:
                    if ent_text == money:
                        continue
                    if ent_type in ("ORG", "PRODUCT"):
                        relations.append((ent_text, "MENTIONS_PRICE", money))
    return relations


# -------------------------
# Main ingestion functions
# -------------------------
def load_all_documents(data_path):
    print(f"📂 Loading documents from: {data_path}")
    documents = []

    if not os.path.exists(data_path):
        print(f"🔴 Directory not found: {data_path}")
        return documents

    for company_dir in os.listdir(data_path):
        company_name = company_dir
        company_path = os.path.join(data_path, company_dir)
        if os.path.isdir(company_path):
            print(f"  -> Loading for company: {company_name}")
            loader = DirectoryLoader(
                company_path,
                loader_cls=UnstructuredLoader,
                recursive=True,
                show_progress=True,
                use_multithreading=True,
                silent_errors=True,
            )
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata["company_name"] = company_name
                doc.metadata["source"] = os.path.basename(doc.metadata.get("source", "unknown"))
            documents.extend(loaded_docs)
            print(f"  -> Loaded {len(loaded_docs)} documents for {company_name}")

    print(f"\n✅ Total documents loaded: {len(documents)}")
    return documents


def split_documents(documents):
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks


def ingest_to_chroma(chunks, db_path):
    if os.path.exists(db_path):
        print(f"🧹 Cleaning old database at: {db_path}")
        shutil.rmtree(db_path)

    print("🧠 Initializing HuggingFace embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"🚀 Ingesting {len(chunks)} chunks into ChromaDB...")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    print("✅ Ingestion complete and database persisted.")
    return db


def enrich_graph_for_documents(documents):
    print("🕸️ Enriching Neo4j knowledge graph from documents...")
    with driver.session() as session:
        for doc in documents:
            company = sanitize(doc.metadata.get("company_name", "unknown"))
            try:
                session.write_transaction(create_company_if_not_exists, company)
                text = doc.page_content
                entities = extract_entities_from_text(text)
                # create entities and mention relationships
                for ent_name, ent_type in entities:
                    name = sanitize(ent_name)
                    if not name:
                        continue
                    session.write_transaction(create_entity_if_not_exists, name, ent_type)
                    session.write_transaction(create_mention_relationship, company, name)
                # infer cheap relationships (like price mentions)
                relations = infer_relationships_from_text(text, entities)
                for subj, rel, obj in relations:
                    s = sanitize(subj)
                    o = sanitize(obj)
                    if s and o:
                        session.write_transaction(create_entity_if_not_exists, s, "INFERRED")
                        session.write_transaction(create_entity_if_not_exists, o, "INFERRED")
                        # use uppercase label safe name
                        rel_label = rel.replace(" ", "_")
                        # create the relationship
                        session.write_transaction(create_simple_relation, s, rel_label, o)
            except Exception as e:
                print(f"  -> Graph enrichment error for doc {doc.metadata.get('source')}: {e}")
    print("✅ Graph enrichment finished.")


def main():
    print("🚀 Starting CI Swarm Ingestion Pipeline (v3 - Graph Enhanced)...")

    documents = load_all_documents(DATA_PATH)
    if not documents:
        print("🔴 No documents found. Exiting.")
        return

    print("🧹 Filtering complex metadata (like 'languages')...")
    filtered_documents = filter_complex_metadata(documents)

    chunks = split_documents(filtered_documents)

    ingest_to_chroma(chunks, DB_PATH)

    # Build/enrich graph from the same filtered docs
    enrich_graph_for_documents(filtered_documents)

    print("\n🎉 --- Pipeline Finished Successfully (RAG + Graph) --- 🎉")
    print(f"📦 Vector DB at '{DB_PATH}' | 🕸️ Graph DB at '{NEO4J_URI}'")


if __name__ == "__main__":
    main()
