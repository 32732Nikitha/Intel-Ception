import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_PATH = "db"
OUR_COMPANY_NAME = "us_tech_nova"
OUTPUT_DIR = "battle_cards"

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Embeddings + DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

# LLM (Gemini fallback)
try:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
except Exception:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

BATTLE_CARD_PROMPT = """
You are a world-class Competitive Intelligence Analyst (Level 3).
Use only the provided context and graph facts.

COMPETITOR: {competitor_name}

DOCUMENT CONTEXT:
{context}

GRAPH FACTS (entities and relationships for this competitor):
{graph_data}

### Tasks:
1) List top 3 strengths (concise bullet points).
2) List top 3 weaknesses.
3) Craft 3 concrete sales talking points to beat this competitor.
4) List source filenames referenced.

Format the battle card clearly and start with a 1-line TL;DR.
"""

prompt_template = ChatPromptTemplate.from_template(BATTLE_CARD_PROMPT)


def query_graph_for_company(company_key: str, limit: int = 30):
    """
    Returns a concise textual list of entities and relationships for the given company_key
    (company_key should be 'competitor_xxx' or the exact company name in the graph).
    """
    # Neo4j stores Company nodes by the company folder name used during ingestion.
    result_lines = []
    with driver.session() as session:
        # Get entities mentioned by company
        try:
            records = session.run(
                "MATCH (c:Company {name:$company})-[:MENTIONS]->(e:Entity) "
                "RETURN e.name AS name, e.type AS type LIMIT $limit",
                company=company_key,
                limit=limit
            )
            for rec in records:
                result_lines.append(f"- {rec['name']} ({rec['type']})")
        except Exception:
            # no results or mismatch; try without prefix
            try:
                records = session.run(
                    "MATCH (c:Company)-[:MENTIONS]->(e:Entity) WHERE toLower(c.name) CONTAINS toLower($company) "
                    "RETURN c.name AS company, e.name AS name, e.type AS type LIMIT $limit",
                    company=company_key.replace("competitor_", ""),
                    limit=limit
                )
                for rec in records:
                    result_lines.append(f"- {rec['name']} ({rec['type']}) [by {rec['company']}]")
            except Exception:
                pass

        # Get simple relations involving those entities (a few)
        try:
            rels = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "WHERE toLower(a.name) CONTAINS toLower($company) OR toLower(b.name) CONTAINS toLower($company) "
                "RETURN a.name AS a, type(r) AS rel, b.name AS b LIMIT 50",
                company=company_key.replace("competitor_", "")
            )
            for r in rels:
                result_lines.append(f"- RELATION: {r['a']} -[{r['rel']}]-> {r['b']}")
        except Exception:
            pass

    if not result_lines:
        return "No graph facts found for this company."
    return "\n".join(result_lines)


# --- Retrieval that uses MMR and expanded fetch_k for diversity ---
def get_relevant_documents(competitor_name):
    competitor_name_str = str(competitor_name).lower().strip()
    competitor_key = f"competitor_{competitor_name_str}"
    print(f"Retrieving documents for: {competitor_key} and {OUR_COMPANY_NAME}")

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 7,
            "filter": {"company_name": competitor_key},
            "fetch_k": 50
        }
    )

    competitor_docs = retriever.invoke(
        f"{competitor_name_str} pricing, features, reviews, and recent press releases"
    )

    our_docs = db.similarity_search(
        query="Tech Nova CRM features, advantages, and strategies",
        k=3,
        filter={"company_name": OUR_COMPANY_NAME}
    )

    print(f"  -> Found {len(competitor_docs)} competitor docs.")
    print(f"  -> Found {len(our_docs)} internal docs.")

    all_docs = competitor_docs + our_docs
    context_str = "\n\n---\n\n".join(
        [f"Source: {doc.metadata.get('source', 'N/A')}\n\n{doc.page_content}" for doc in all_docs]
    )
    sources = list(dict.fromkeys([doc.metadata.get('source', 'N/A') for doc in all_docs]))
    return context_str, sources


def get_context(inputs):
    competitor_name = inputs["competitor_name"]
    context, sources = get_relevant_documents(competitor_name)
    # fetch graph facts for the company key used in ingestion
    graph_data = query_graph_for_company(f"competitor_{competitor_name.lower().strip()}")
    return {
        "context": context,
        "sources": ", ".join(sources),
        "graph_data": graph_data
    }


# Build RAG chain
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"competitor_name": RunnablePassthrough()}
    | RunnablePassthrough.assign(context_and_sources=get_context)
    | RunnablePassthrough.assign(
        competitor_name=lambda x: x["competitor_name"],
        context=lambda x: x["context_and_sources"]["context"],
        graph_data=lambda x: x["context_and_sources"]["graph_data"],
        sources=lambda x: x["context_and_sources"]["sources"]
    )
    | prompt_template
    | llm
    | StrOutputParser()
)


def generate_battle_card(competitor_name: str):
    competitor_name = competitor_name.lower().strip().replace("competitor_", "")
    print(f"Generating graph-enhanced battle card for: {competitor_name}")
    try:
        response = rag_chain.invoke(competitor_name)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        file_path = os.path.join(OUTPUT_DIR, f"{competitor_name}_battle_card_v3.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"✅ Saved battle card: {file_path}")
        return response
    except Exception as e:
        print(f"Error generating battle card for {competitor_name}: {e}")
        return None


def main():
    print("Analyst 2.0 active — generating all graph-aware battle cards...")
    # detect competitors in DB
    try:
        collections = db.get()
        all_companies = list(set([
            meta.get("company_name")
            for meta in collections["metadatas"]
            if meta.get("company_name") and meta.get("company_name") != OUR_COMPANY_NAME
        ]))
        all_companies = [c.replace("competitor_", "") for c in all_companies]
    except Exception as e:
        print(f"Could not auto-detect competitors: {e}")
        all_companies = []

    if not all_companies:
        print("No competitors detected in the vector database.")
        return

    print("Detected competitors:", all_companies)
    for name in all_companies:
        generate_battle_card(name)


if __name__ == "__main__":
    main()
