import streamlit as st
import os
import time
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from neo4j import GraphDatabase
from pyvis.network import Network
import tempfile
from dotenv import load_dotenv

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="🧠 Intel-Ception v3 — Graph-Enhanced CI",
    page_icon="⚔️",
    layout="wide"
)

INTAKE_PATH = "_intake"
BATTLE_CARDS_PATH = "battle_cards"
os.makedirs(INTAKE_PATH, exist_ok=True)
os.makedirs(BATTLE_CARDS_PATH, exist_ok=True)

# =====================================================
# LOAD ENVIRONMENT VARIABLES & CONNECT TO NEO4J
# =====================================================
load_dotenv()  # Load from .env file in project root

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    st.error("❌ Missing Neo4j credentials! Please check your .env file.")
    st.stop()

# Attempt to connect to AuraDB
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("RETURN 1")
    st.success("✅ Connected to Neo4j AuraDB successfully!")
except Exception as e:
    st.error(f"❌ Failed to connect to Neo4j: {e}")
    st.stop()

# =====================================================
# STYLE CONFIGURATION
# =====================================================
st.markdown("""
<style>
html, body, [class*="css"] {
   font-family:'Helvetica Neue','Helvetica','Arial',sans-serif;
   background-color:#0E1117;
   color:white;
}
h1 {color:#4A90E2;font-weight:700;}
h2 {color:#50E3C2;border-bottom:1px solid #3c3c4f;padding-bottom:4px;}
div[data-testid="stVerticalBlock"]>div[data-testid="stHorizontalBlock"]>div[data-testid="stVerticalBlock"]{
    background-color:#1A1A2E;border:1px solid #3c3c4f;border-radius:10px;padding:20px;
}
button[kind="secondary"]{
    border-radius:8px;
    background:#4A90E2;
    color:white;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# PDF GENERATOR FUNCTION
# =====================================================
def generate_pdf(text_content: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    for line in text_content.split("\n"):
        line = line.strip()
        if line.startswith("### "):
            pdf.set_font("Helvetica", 'B', 16)
            pdf.multi_cell(0, 10, line.replace("### ", ""), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=10)
        elif line:
            pdf.multi_cell(0, 5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        else:
            pdf.ln(5)
    return bytes(pdf.output(dest="S"))

# =====================================================
# HEADER
# =====================================================
st.title("⚔️ Intel-Ception v3 — Analyst 2.0")
st.caption("Graph-Enhanced Competitive Intelligence Dashboard | Powered by Tech Nova")

tab1, tab2, tab3 = st.tabs(["📤 Upload Intel", "📊 Battle Cards", "🕸 Knowledge Graph"])

# =====================================================
# TAB 1 — UPLOAD INTEL
# =====================================================
with tab1:
    st.header("📥 Upload New Intel")
    company_name = st.text_input("Competitor Folder Name", placeholder="competitor_zoho").strip().lower()
    uploaded_files = st.file_uploader("Upload TXT / PDF / DOCX", accept_multiple_files=True, type=['txt', 'pdf', 'docx', 'md'])

    if st.button("💾 Upload Files"):
        if not company_name:
            st.error("Enter competitor name.")
        elif not uploaded_files:
            st.warning("Select files to upload.")
        else:
            company_dir = os.path.join(INTAKE_PATH, company_name)
            os.makedirs(company_dir, exist_ok=True)
            for f in uploaded_files:
                with open(os.path.join(company_dir, f.name), "wb") as out:
                    out.write(f.getbuffer())
            st.success(f"✅ Saved {len(uploaded_files)} file(s) to {company_dir}")
            st.toast("Files uploaded successfully!")

    st.info("""
**Agent Pipeline v3**
- `proactive_scout.py` watches _intake/ and detects major changes.  
- `analyst_agent.py` now uses Neo4j Graph + RAG for deep insights.  
- Each battle card combines semantic and graph reasoning 🧠.
""")

# =====================================================
# TAB 2 — BATTLE CARDS VIEW
# =====================================================
with tab2:
    st.header("📊 Generated Battle Cards")

    def list_cards():
        return sorted(
            [f for f in os.listdir(BATTLE_CARDS_PATH) if f.endswith(".txt")],
            key=lambda x: os.path.getmtime(os.path.join(BATTLE_CARDS_PATH, x)),
            reverse=True
        )

    if st.button("🔄 Refresh List"):
        st.cache_data.clear()

    @st.cache_data(ttl=10)
    def get_cards():
        return list_cards()

    cards = get_cards()
    if not cards:
        st.warning("No battle cards yet. Run Analyst 2.0 after ingestion.")
    else:
        selected = st.selectbox("Select Battle Card", cards)
        file_path = os.path.join(BATTLE_CARDS_PATH, selected)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        st.caption(f"Last Updated: {time.ctime(os.path.getmtime(file_path))}")
        with st.expander(f"🧾 {selected}", expanded=True):
            # Clean and format Markdown content
            formatted_content = content.replace("###", "##").replace("**", "**").replace("•", "-")
            st.markdown(formatted_content, unsafe_allow_html=True)

        pdf_bytes = generate_pdf(content)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ PDF", pdf_bytes, file_name=selected.replace(".txt", ".pdf"), mime="application/pdf")
        with c2:
            st.download_button("⬇️ TXT", content, file_name=selected, mime="text/plain")

# =====================================================
# TAB 3 — KNOWLEDGE GRAPH VISUALIZATION
# =====================================================
with tab3:
    st.header("🕸 Knowledge Graph Explorer (Neo4j)")
    st.caption("Visualize entities and relationships extracted from competitor documents.")

    company_query = st.text_input("Enter Company Name (e.g. competitor_zoho)").strip()
    limit = st.slider("Limit nodes", 10, 100, 30)

    if st.button("🔍 Generate Graph") and company_query:
        try:
            with driver.session() as session:
                records = list(session.run("""
                    MATCH (c:Company {name:$company})-[:MENTIONS]->(e:Entity)
                    OPTIONAL MATCH (e)-[r]->(x:Entity)
                    RETURN c.name AS company, e.name AS entity, type(r) AS rel, x.name AS target LIMIT $limit
                """, company=company_query, limit=limit))

                if not records:
                    st.warning("⚠️ No matching data found in Neo4j for this company. Try another name.")
                else:
                    net = Network(height="600px", width="100%", bgcolor="#0E1117", font_color="white")
                    net.barnes_hut()
                    added = set()

                    for rec in records:
                        c, e, r, t = rec["company"], rec["entity"], rec["rel"], rec["target"]
                        if c and c not in added:
                            net.add_node(c, label=c, color="#4A90E2")
                            added.add(c)
                        if e and e not in added:
                            net.add_node(e, label=e, color="#50E3C2")
                            added.add(e)
                        net.add_edge(c, e, label="MENTIONS")
                        if t and r:
                            if t not in added:
                                net.add_node(t, label=t, color="#F5A623")
                                added.add(t)
                            net.add_edge(e, t, label=r)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
                        net.save_graph(tmp.name)
                        with open(tmp.name, "r", encoding="utf-8") as html_file:
                            html = html_file.read()
                        st.components.v1.html(html, height=620, scrolling=True)

        except Exception as e:
            st.error(f"❌ Neo4j query failed: {e}")

    st.info("""
**Legend 🔹**
- Blue = Company  
- Green = Entities mentioned  
- Orange = Related entities (e.g., pricing, products)  
""")
