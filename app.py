
import os
import streamlit as st

from crewai import Agent, Task, Crew, LLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Correct imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Marketing Campaign Builder",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Autonomous AI Marketing Campaign Builder (11 Agents)")
st.markdown("Generate complete marketing campaigns using AI Agents + Gemini")

# =========================
# GEMINI API KEY
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found!")
    st.stop()

# =========================
# GEMINI MODEL
# =========================
llm = LLM(
    model="gemini/gemma-4-31b-it",
    temperature=0.7,
    api_key=GEMINI_API_KEY,
    max_retries=6
)

# =========================
# VECTOR DATABASE
# =========================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    persist_db_path = "./chroma_db"
    vectorstore = Chroma(
        persist_directory=persist_db_path,
        embedding_function=embeddings
    )
    return vectorstore

vectorstore = load_vectorstore()

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Campaign Inputs")

brand_name = st.sidebar.text_input("Brand Name", "Nike")
product_name = st.sidebar.text_input("Product / Service", "AI Fitness Shoes")
target_audience = st.sidebar.text_input("Target Audience", "College Students")

marketing_goal = st.sidebar.selectbox(
    "Marketing Goal",
    ["Brand Awareness", "Lead Generation", "Sales Conversion", "Product Launch"]
)

campaign_budget = st.sidebar.text_input("Campaign Budget", "$5000")

platforms = st.sidebar.multiselect(
    "Platforms",
    ["Instagram", "LinkedIn", "Twitter", "YouTube", "Facebook", "TikTok", "Telegram", "Email"],
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
for key in ["campaign_result", "brand", "product", "audience", "goal", "budget", "selected_platforms"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "campaign_result" else ([] if key == "selected_platforms" else "")

# =========================
# GENERATE CAMPAIGN
# =========================
if st.button("Generate Campaign"):
    with st.spinner("AI Agents are building your campaign (11 agents running)..."):

        # =========================
        # RAG CONTEXT
        # =========================
        rag_query = f"Marketing strategies for {brand_name} product {product_name} audience {target_audience}"
        rag_context = retrieve_context(rag_query)

        # =========================
        # AGENTS
        # =========================
        researcher = Agent(
            role="Trend Researcher",
            goal=f"Analyze current marketing trends, tactics, and strategies relevant to the campaign for brand {brand_name}",
            backstory="You are an expert market analyst who specializes in uncovering low-cost, high-impact growth channels for bootstrapped startups.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        competitor_analyst = Agent(
            role="Competitor Intelligence Analyst",
            goal=f"Analyze competitor positioning, marketing angles, strengths, and weaknesses for {brand_name}",
            backstory="You are a strategic intelligence specialist. You study competitor landing pages and messaging to identify gaps and differentiation opportunities.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        seo_specialist = Agent(
            role="SEO Specialist",
            goal=f"Construct a high-performance SEO strategy for {brand_name}'s product {product_name}",
            backstory="You are a data-driven SEO wizard. You excel at search intent analysis, keyword structuring, and building topical authority plans.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        content_strategist = Agent(
            role="Content Strategist",
            goal=f"Create a structured campaign strategy and content map for {brand_name}",
            backstory="You are a creative campaign planner. You synthesize raw trend data and competitor analysis into high-level content pillars and content calendars.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        writer = Agent(
            role="Content Writer",
            goal=f"Draft an engaging, authoritative, and direct blog article for brand {brand_name}. Adhere to writing voice guidelines.",
            backstory="You are an elite copywriter trained in direct-response writing. You avoid buzzwords at all costs and write punchy, conversational articles.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        social_manager = Agent(
            role="Social Media Manager",
            goal=f"Translate the written campaign assets into high-performance social posts for the selected platforms: {', '.join(platforms)}.",
            backstory="You are a social-native storyteller. You know how to hook attention on LinkedIn and structure Twitter threads.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        email_marketer = Agent(
            role="Email Marketer",
            goal=f"Draft direct-response email sequences and newsletters for {brand_name} that build trust and drive conversions.",
            backstory="You are an expert email marketer. You focus on personalization, clear subject lines, and compelling CTAs.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        campaign_planner = Agent(
            role="Campaign Planner",
            goal=f"Structure a detailed campaign launch timeline and platform selection blueprint for {brand_name} with budget {campaign_budget}.",
            backstory="You are an organized campaign coordinator. You determine where and when content should be distributed to maximize organic reach.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        performance_analyst = Agent(
            role="Performance Analytics Specialist",
            goal=f"Establish a complete measurement framework, KPIs, and tracking metrics for the {brand_name} campaign.",
            backstory="You are a data-driven growth analyst. You define exactly how to measure awareness, engagement, conversion, and retention.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        brand_compliance_officer = Agent(
            role="Brand Compliance Officer",
            goal=f"Audit all generated campaign assets for brand consistency, compliance, and ensure no prohibited buzzwords are used.",
            backstory="You are a strict compliance auditor and editor. You verify final outputs against brand positioning, tone of voice, and compliance rules.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        executive_reporter = Agent(
            role="Executive Reporter",
            goal=f"Compile and format all inputs from the agents into a single, cohesive, premium marketing campaign report for {brand_name}.",
            backstory="You are a professional technical communicator. You structure complex strategy documents into readable executive summaries and markdown reports.",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )

        # =========================
        # TASKS
        # =========================
        research_task = Task(
            description=f"Perform detailed market research. Brand: {brand_name}, Product: {product_name}, Audience: {target_audience}, Goal: {marketing_goal}. Context: {rag_context}. Include: audience insights, current trends, and growth opportunities.",
            expected_output="Detailed marketing research report.",
            agent=researcher
        )

        competitor_task = Task(
            description=f"Analyze key competitors for {brand_name} in the {product_name} space. Identify their value propositions, marketing channels, and weaknesses.",
            expected_output="A competitive analysis report with 3 key differentiation opportunities.",
            agent=competitor_analyst
        )

        seo_task = Task(
            description=f"Develop a target SEO strategy for {brand_name}'s {product_name} targeting {target_audience}. Provide: 1 primary keyword, 3-5 LSI keywords, an SEO title, a meta description under 160 characters, and a H2/H3 outline.",
            expected_output="An SEO strategy blueprint containing target keywords and structured outline.",
            agent=seo_specialist
        )

        strategist_task = Task(
            description=f"Create a content strategy for {brand_name}. Define 3 primary content pillars and map them to customer awareness stages.",
            expected_output="A content strategy map detailing 3 content pillars, awareness stage alignments, and target messaging.",
            agent=content_strategist
        )

        writing_task = Task(
            description=f"Write a complete, high-quality, long-form blog article for {brand_name}'s {product_name} targeting {target_audience}. Follow the SEO and Content Strategy outlines.",
            expected_output="A complete, ready-to-publish blog article (700-1000 words) in a bold, conversational voice.",
            agent=writer
        )

        social_task = Task(
            description=f"Review the generated blog article and write social media promotions for platforms: {', '.join(platforms)}. Create 1 LinkedIn post using the 'hook -> story -> value -> CTA' structure and a Twitter/X thread of 4-6 tweets.",
            expected_output="A social media package including 1 LinkedIn post and 1 Twitter/X thread.",
            agent=social_manager
        )

        email_task = Task(
            description=f"Draft an email marketing package for {brand_name} consisting of 1 welcome email and a 2-part educational newsletter sequence on the topic of {product_name}.",
            expected_output="Complete email marketing copy sequence with subject lines, body copy, and CTAs.",
            agent=email_marketer
        )

        planning_task = Task(
            description=f"Develop a 4-week distribution calendar for {brand_name}'s {marketing_goal} campaign across platforms: {', '.join(platforms)}. Budget: {campaign_budget}.",
            expected_output="A structured 4-week campaign calendar and platform distribution schedule.",
            agent=campaign_planner
        )

        analytics_task = Task(
            description=f"Design a performance measurement framework for {brand_name}'s campaign. List specific metrics for Awareness, Engagement, Conversion, and Retention categories.",
            expected_output="A performance measurement matrix detailing KPIs, target benchmarks, and tracking methods.",
            agent=performance_analyst
        )

        compliance_task = Task(
            description=f"Audit all generated campaign assets for {brand_name} against brand guidelines, tone of voice rules, and compliance policies. Identify and flag any prohibited buzzwords.",
            expected_output="A brand compliance audit report highlighting issues found and corrective actions taken.",
            agent=brand_compliance_officer
        )

        reporting_task = Task(
            description=f"Synthesize the outputs of all 10 previous tasks for {brand_name} into a single comprehensive, executive-ready final marketing campaign report with a clear table of contents.",
            expected_output="A cohesive, end-to-end markdown report compiling the entire campaign package.",
            agent=executive_reporter
        )

        # =========================
        # CREW AI - ALL 11 AGENTS
        # =========================
        crew = Crew(
            agents=[
                researcher,
                competitor_analyst,
                seo_specialist,
                content_strategist,
                writer,
                social_manager,
                email_marketer,
                campaign_planner,
                performance_analyst,
                brand_compliance_officer,
                executive_reporter
            ],
            tasks=[
                research_task,
                competitor_task,
                seo_task,
                strategist_task,
                writing_task,
                social_task,
                email_task,
                planning_task,
                analytics_task,
                compliance_task,
                reporting_task
            ],
            verbose=True
        )

        # =========================
        # RUN CREW
        # =========================
        result = crew.kickoff()

        # Save results to Session State to persist across reruns
        st.session_state["campaign_result"] = str(result)
        st.session_state["brand"] = brand_name
        st.session_state["product"] = product_name
        st.session_state["audience"] = target_audience
        st.session_state["goal"] = marketing_goal
        st.session_state["budget"] = campaign_budget
        st.session_state["selected_platforms"] = platforms

# =========================
# DISPLAY OUTPUT & N8N AUTOMATION PIPELINE (Outside the button click block)
# =========================
if st.session_state["campaign_result"] is not None:
    st.success("Campaign Generated Successfully!")
    st.markdown("## AI Campaign Report")
    st.write(st.session_state["campaign_result"])

    # =========================
    # DOWNLOAD REPORT
    # =========================
    st.download_button(
        label="Download Campaign Report",
        data=st.session_state["campaign_result"],
        file_name="marketing_campaign_report.txt",
        mime="text/plain"
    )

    # =========================
    # N8N PIPELINE INTEGRATION
    # =========================
    st.markdown("---")
    st.markdown("### Automation Pipeline (n8n)")

    n8n_url = st.text_input(
        "Enter your n8n Production Webhook URL",
        placeholder="https://nithya-sri-14.app.n8n.cloud/webhook-test/crewai-campaign-trigger"
    )

    if st.button("Trigger n8n Campaign Pipeline"):
        if not n8n_url:
            st.warning("Please provide a valid n8n Webhook URL!")
        else:
            with st.spinner("Pushing campaign payloads to n8n..."):
                try:
                    import requests
                    payload = {
                        "brand_name": st.session_state["brand"],
                        "product_name": st.session_state["product"],
                        "target_audience": st.session_state["audience"],
                        "marketing_goal": st.session_state["goal"],
                        "campaign_budget": st.session_state["budget"],
                        "platforms": st.session_state["selected_platforms"],
                        "campaign_report": st.session_state["campaign_result"]
                    }
                    response = requests.post(n8n_url, json=payload, timeout=30)
                    if response.status_code == 200:
                        st.success("Campaign successfully sent to n8n workflow!")
                    else:
                        st.error(f"Failed to trigger n8n (HTTP {response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Webhook Error: {str(e)}")
