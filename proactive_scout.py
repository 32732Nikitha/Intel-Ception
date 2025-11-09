import os
import shutil
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# import our analyst engine
from analyst_agent import generate_battle_card

# filter helper for metadata
from langchain_community.vectorstores.utils import filter_complex_metadata

load_dotenv()

INTAKE_PATH = "_intake"
DATA_PATH = "ci_data_lake"
DB_PATH = "db"
CHECK_INTERVAL_SECONDS = 30

# Initialize embeddings + LLM + DB connection
try:
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Loading Gemini model for change detection...")
    try:
        llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    except Exception:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    print("Connecting to Chroma DB...")
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    print("Components initialized.")
except Exception as e:
    print(f"Fatal init error: {e}")
    raise

CHANGELOG_PROMPT_TEMPLATE = """
You are a B2B Competitive Intelligence Analyst. Your job is to detect major, actionable changes
in competitor documents.

You will be given the OLD context we already have on a competitor, and a NEW document we just found.
Compare them and determine if the NEW document contains a "major strategic change."

A "major strategic change" is defined as:
- A new product announcement or product line.
- A change in pricing, pricing models (e.g., adding credits), or new API limits.
- A new strategic partnership, acquisition, or funding round.
- A significant change in core product features.

---
OLD CONTEXT (What we already know about this competitor):
{old_context}
---
NEW DOCUMENT (The file we just found):
{new_doc_text}
---

Based on this comparison, does the NEW DOCUMENT contain a "major strategic change"?
Answer with a single word: YES or NO.
"""

CHANGELOG_PROMPT = ChatPromptTemplate.from_template(CHANGELOG_PROMPT_TEMPLATE)
DETECTION_CHAIN = CHANGELOG_PROMPT | llm | StrOutputParser()


def check_for_major_change(new_doc, company_name):
    print(f"Analysing '{new_doc.metadata.get('source')}' for major changes...")
    retriever = db.as_retriever(search_kwargs={"filter": {"company_name": company_name}, "k": 5})
    old_docs = retriever.invoke(f"existing info on {company_name} pricing, features, and products")
    old_context = "\n---\n".join([d.page_content for d in old_docs])

    if not old_docs:
        print("-> No old context found. Flagging as major change.")
        return True

    response = DETECTION_CHAIN.invoke({
        "old_context": old_context,
        "new_doc_text": new_doc.page_content
    })
    print(f"-> LLM decision: {response}")
    return "YES" in response.upper()


def ingest_new_document(doc, db_connection):
    """
    Splits and ingests a single new document into the persistent ChromaDB
    and triggers graph enrichment via the top-level ingestion pipeline (re-using ingest_data).
    """
    print(f"-> Ingesting '{doc.metadata.get('source')}' into database...")

    # Filter complex metadata first
    filtered_docs = filter_complex_metadata([doc])
    if not filtered_docs:
        print("-> Document filtered out as empty metadata.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(filtered_docs)

    if not chunks:
        print("-> Warning: Document was empty after splitting.")
        return

    db_connection.add_documents(chunks)
    print(f"-> Added {len(chunks)} chunks to the vector DB.")


def move_processed_file(file_path, company_name):
    try:
        filename = os.path.basename(file_path)
        destination_dir = os.path.join(DATA_PATH, company_name)
        os.makedirs(destination_dir, exist_ok=True)
        destination_path = os.path.join(destination_dir, filename)
        shutil.move(file_path, destination_path)
        print(f"-> Moved file to: {destination_path}")
    except Exception as e:
        print(f"Error moving file {file_path}: {e}")


def scan_intake_folder():
    print(f"\nScanning for new documents in: {INTAKE_PATH}...")
    new_files_found = 0

    for company_dir, _, filenames in os.walk(INTAKE_PATH):
        if not filenames:
            continue

        company_name = os.path.basename(company_dir)
        if company_name == INTAKE_PATH:
            continue

        print(f"Found new files for company: {company_name}")
        for filename in filenames:
            new_files_found += 1
            file_path = os.path.join(company_dir, filename)
            print(f"--- Processing: {filename} ---")

            try:
                loader = UnstructuredFileLoader(file_path)
                loaded = loader.load()
                if not loaded:
                    print("-> Loader returned no content. Skipping file.")
                    move_processed_file(file_path, company_name)
                    continue
                new_doc = loaded[0]
                new_doc.metadata["company_name"] = company_name
                new_doc.metadata["source"] = filename

                is_major_change = check_for_major_change(new_doc, company_name)

                if is_major_change:
                    print("\nMAJOR STRATEGIC CHANGE DETECTED!")
                    print(f"-> Document '{filename}' contains a major update.")
                    print("-> ACTION: Triggering Analyst to regenerate battle card...")
                    try:
                        generate_battle_card(company_name)
                        print("-> Battle card regenerated.")
                    except Exception as e:
                        print(f"-> Failed to regenerate battle card: {e}")

                else:
                    print("\nNo major change detected. Ingesting for record-keeping.")

                # Ingest into vector DB
                ingest_new_document(new_doc, db)

                # Move file into the permanent data lake
                move_processed_file(file_path, company_name)

                print(f"--- Finished processing: {filename} ---\n")
            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    if new_files_found == 0:
        print("No new documents found.")


def main():
    print(f"\n--- Proactive Scout Agent Activated ---")
    print(f"--- Watching folder: {INTAKE_PATH} ---")
    print(f"--- Checking every {CHECK_INTERVAL_SECONDS} seconds. Press Ctrl+C to stop. ---")

    os.makedirs(INTAKE_PATH, exist_ok=True)

    while True:
        try:
            scan_intake_folder()
            print(f"Sleeping for {CHECK_INTERVAL_SECONDS} seconds...")
            time.sleep(CHECK_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nShutting down scout daemon...")
            break
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            print("Restarting loop after 60 seconds...")
            time.sleep(60)


if __name__ == "__main__":
    main()
