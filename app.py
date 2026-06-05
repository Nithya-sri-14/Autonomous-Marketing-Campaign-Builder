import os
import sys

# AWS Compatibility: Patch SQLite3 for ChromaDB and set HF_HOME for writable cache directory
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

os.environ["HF_HOME"] = "/tmp/huggingface"

import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv()

from crewai import Agent, Task, Crew, LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Marketing Campaign Builder",
    page_icon="🚀",
    layout="wide"
)

# =========================
# CUSTOM PREMIUM STYLING
# =========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stSidebar"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Title Gradient */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Styled container card for campaign output */
    .report-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Custom style for buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white !important;
        border: none;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        width: 100%;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
        color: white !important;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        font-weight: 700 !important;
    }
    
    /* Info/Success metrics styling */
    .metric-box {
        background-color: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366f1;
        padding: 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🚀 Autonomous AI Marketing Campaign Builder</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Orchestrate role-playing AI agents and inject RAG-powered brand guidelines to build premium marketing assets.</div>', unsafe_allow_html=True)

# =========================
# GEMINI API KEY
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.warning("⚠️ GEMINI_API_KEY environment variable not found. Please enter it below or configure your .env file.")
    GEMINI_API_KEY = st.text_input("Gemini API Key", type="password")
    if not GEMINI_API_KEY:
        st.info("Please set the Gemini API Key to continue.")
        st.stop()

# Force key propagation to all Google GenAI libraries
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Display key status in sidebar for debugging
st.sidebar.success(f"🔑 API Key: {GEMINI_API_KEY[:6]}... (len: {len(GEMINI_API_KEY)})")

# =========================
# GEMINI MODEL
# =========================
llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
    api_key=GEMINI_API_KEY,
    max_retries=6
)

# =========================
# VECTOR DATABASE (AUTO-BUILD)
# =========================
@st.cache_resource
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004"
    )
    persist_db_path = "./chroma_db"
    
    # Check if the DB path exists and is populated
    if not os.path.exists(persist_db_path) or not os.listdir(persist_db_path):
        brand_path = os.path.join("data", "brand.txt")
        if os.path.exists(brand_path):
            from langchain_community.document_loaders import TextLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            loader = TextLoader(brand_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)
            
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_db_path
            )
        else:
            # Create a basic guidelines file if missing
            os.makedirs("data", exist_ok=True)
            default_guidelines = """[BRAND IDENTITY & VOICE GUIDELINES]
TONE & STYLE:
- Startup Tone: Bold, high-energy, ambitious, transparent, and direct.
- Conversational Writing: Speak directly to the reader. Use 'you' and 'we'.
- Founder-Focused: Speak to resource-constrained builders.
"""
            with open(brand_path, "w", encoding="utf-8") as f:
                f.write(default_guidelines)
            
            from langchain_community.document_loaders import TextLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            loader = TextLoader(brand_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)
            
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=persist_db_path
            )
    else:
        vectorstore = Chroma(
            persist_directory=persist_db_path,
            embedding_function=embeddings
        )
    return vectorstore

try:
    vectorstore = load_vectorstore()
except Exception as e:
    st.error(f"Failed to load/initialize Chroma DB: {e}")
    st.markdown(
        """
        ### 🔍 How to resolve this issue:
        
        1. **Check your Gemini API Key**:
           - Make sure your key is copied completely and starts with `AIzaSy...`.
           - In AWS Elastic Beanstalk, make sure you configured `GEMINI_API_KEY` under **Configuration** -> **Environment properties** (Platform properties).
           
        2. **Enable the Generative Language API**:
           - If you created your API key using the **Google Cloud Console** instead of **Google AI Studio**, you must manually go to your GCP project and **enable the Generative Language API**.
           - If using Google AI Studio, this API is enabled automatically.
           
        3. **Clear Cached Files**:
           - If you previously had a corrupted database build, clear the cache. In AWS, redeploying with a different version label will reset the staging folder.
        """
    )
    st.stop()

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Campaign Inputs")

brand_name = st.sidebar.text_input(
    "Brand Name",
    "Nike"
)

product_name = st.sidebar.text_input(
    "Product / Service",
    "AI Fitness Shoes"
)

target_audience = st.sidebar.text_input(
    "Target Audience",
    "College Students"
)

marketing_goal = st.sidebar.selectbox(
    "Marketing Goal",
    [
        "Brand Awareness",
        "Lead Generation",
        "Sales Conversion",
        "Product Launch"
    ]
)

campaign_budget = st.sidebar.text_input(
    "Campaign Budget",
    "$5000"
)

platforms = st.sidebar.multiselect(
    "Platforms",
    [
        "Instagram",
        "LinkedIn",
        "Twitter",
        "YouTube",
        "Facebook",
        "TikTok",
        "Telegram",
        "Email"
    ],
    default=["Instagram", "Twitter"]
)

# =========================
# RAG FUNCTION
# =========================
def retrieve_context(query):
    try:
        docs = vectorstore.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    except Exception as e:
        return f"RAG Error: {str(e)}"

# =========================
# INITIALIZE SESSION STATE
# =========================
if "campaign_result" not in st.session_state:
    st.session_state["campaign_result"] = None
if "brand" not in st.session_state:
    st.session_state["brand"] = ""
if "product" not in st.session_state:
    st.session_state["product"] = ""
if "audience" not in st.session_state:
    st.session_state["audience"] = ""
if "goal" not in st.session_state:
    st.session_state["goal"] = ""
if "budget" not in st.session_state:
    st.session_state["budget"] = ""
if "selected_platforms" not in st.session_state:
    st.session_state["selected_platforms"] = []

# =========================
# GENERATE CAMPAIGN
# =========================
if st.button("Generate Campaign"):
    with st.spinner("AI Agents are building your campaign..."):
        # Retrieve guidelines context
        rag_query = f"Marketing guidelines and tone for {brand_name} {product_name}"
        rag_context = retrieve_context(rag_query)

        # Initialize Agents
        market_research_agent = Agent(
            role="Market Research Analyst",
            goal=f"Analyze market trends, audience behavior, competitors, and opportunities for {brand_name}.",
            backstory="You are an expert marketing strategist specializing in audience research and competitive analysis.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        content_agent = Agent(
            role="Content Strategist",
            goal="Create engaging social media and campaign content.",
            backstory="You are a viral content creator with expertise in modern marketing.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        ad_agent = Agent(
            role="Advertising Expert",
            goal="Create paid ad strategy, budget allocation, and conversion ideas.",
            backstory="You are a digital ads specialist skilled in high-converting campaigns.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        # Initialize Tasks
        research_task = Task(
            description=f"""Perform detailed market research.
Brand: {brand_name}
Product: {product_name}
Audience: {target_audience}
Goal: {marketing_goal}

Use this additional brand guidelines context:
{rag_context}

Include:
- Audience insights
- Competitor analysis
- Current trends
- Opportunities""",
            expected_output="Detailed marketing research report.",
            agent=market_research_agent
        )

        content_task = Task(
            description=f"""Create a content strategy.
Platforms: {', '.join(platforms)}

Include:
- Social media post ideas
- Content calendar
- Captions
- Hashtag strategy
- Viral campaign ideas""",
            expected_output="Complete content marketing plan.",
            agent=content_agent
        )

        ad_task = Task(
            description=f"""Create a paid advertising strategy.
Budget: {campaign_budget}

Include:
- Ad targeting
- Budget allocation
- Best platforms
- Conversion strategy
- KPI recommendations""",
            expected_output="Complete paid ads strategy.",
            agent=ad_agent
        )

        # Assemble Crew
        crew = Crew(
            agents=[market_research_agent, content_agent, ad_agent],
            tasks=[research_task, content_task, ad_task],
            verbose=True
        )

        # Run Crew
        result = crew.kickoff()

        # Save to Session State
        st.session_state["campaign_result"] = str(result)
        st.session_state["brand"] = brand_name
        st.session_state["product"] = product_name
        st.session_state["audience"] = target_audience
        st.session_state["goal"] = marketing_goal
        st.session_state["budget"] = campaign_budget
        st.session_state["selected_platforms"] = platforms

# =========================
# DISPLAY OUTPUT & N8N AUTOMATION PIPELINE
# =========================
if st.session_state["campaign_result"] is not None:
    st.success("Campaign Generated Successfully!")
    
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown("## 📊 AI Campaign Report")
    st.markdown(st.session_state["campaign_result"])
    st.markdown('</div>', unsafe_allow_html=True)

    # Download Button
    st.download_button(
        label="Download Campaign Report",
        data=st.session_state["campaign_result"],
        file_name=f"{st.session_state['brand'].lower()}_campaign_report.md",
        mime="text/markdown"
    )
