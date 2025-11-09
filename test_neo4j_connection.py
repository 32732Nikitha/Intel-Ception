from neo4j import GraphDatabase
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get credentials from .env file
URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PASSWORD = os.getenv("NEO4J_PASSWORD")

print(f"🔗 Trying to connect to: {URI}")

# Try connecting
try:
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    print("✅ Connected to Neo4j AuraDB successfully!")
except Exception as e:
    print("❌ Connection failed:", e)
finally:
    driver.close()
