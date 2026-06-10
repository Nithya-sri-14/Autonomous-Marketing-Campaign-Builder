import os
import sys
import argparse
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# AWS Compatibility: Patch SQLite3 for ChromaDB and set HF_HOME for writable cache directory
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

os.environ["HF_HOME"] = "/tmp/huggingface"

# Apply nest_asyncio to allow asyncio to run in already-running event loops
nest_asyncio.apply()

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    persist_db_path = "./chroma_db"
    
    # Auto-create vectorstore from guidelines if it doesn't exist
    if not os.path.exists(persist_db_path) or not os.listdir(persist_db_path):
        brand_path = os.path.join("data", "brand.txt")
        if not os.path.exists(brand_path):
            os.makedirs("data", exist_ok=True)
            default_guidelines = """[BRAND IDENTITY & VOICE GUIDELINES]

TONE & STYLE:
- Startup Tone: Bold, high-energy, ambitious, transparent, and direct.
- Conversational Writing: Speak directly to the reader. Use 'you' and 'we'. Avoid complex corporate sentences.
- Founder-Focused: Speak to resource-constrained builders, bootstrappers, and startup builders.
- Layout: Use short paragraphs (1-3 sentences max) to ensure high readability on mobile and web screens.
- Emojis: Integrate relevant emojis strategically to break up text, but avoid over-cluttering.
- SEO-Friendly Structure: Lead with key takeaways, use rich and logical H2/H3 heading hierarchies.

PROHIBITED FLUFF & BUZZWORDS (DO NOT USE UNDER ANY CIRCUMSTANCES):
- 'delve'
- 'testament'
- 'revolutionize'
- 'tapestry'
- 'moreover'
- 'furthermore'
- 'it is crucial to remember'
"""
            with open(brand_path, "w", encoding="utf-8") as f:
                f.write(default_guidelines)
            print("📝 Created default brand voice guidelines at data/brand.txt")
            
        print("📦 Indexing brand guidelines in vector database...")
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
        print("✅ Vector database populated successfully!")
    else:
        vectorstore = Chroma(
            persist_directory=persist_db_path,
            embedding_function=embeddings
        )
    return vectorstore

def main():
    parser = argparse.ArgumentParser(description="Autonomous Marketing Campaign Builder (CLI Version - 11 Agents)")
    parser.add_argument(
        "--topic", 
        type=str, 
        default="AI-driven organic marketing for bootstrapped startups",
        help="Campaign topic for the multi-agent system"
    )
    parser.add_argument(
        "--brand", 
        type=str, 
        default="Nike",
        help="Brand name for the campaign"
    )
    parser.add_argument(
        "--product", 
        type=str, 
        default="AI Fitness Shoes",
        help="Product name for the campaign"
    )
    parser.add_argument(
        "--audience", 
        type=str, 
        default="College Students",
        help="Target audience for the campaign"
    )
    parser.add_argument(
        "--goal", 
        type=str, 
        default="Brand Awareness",
        help="Marketing goal for the campaign"
    )
    parser.add_argument(
        "--budget", 
        type=str, 
        default="$5000",
        help="Campaign budget"
    )
    parser.add_argument(
        "--platforms", 
        type=str, 
        default="Instagram, Twitter",
        help="Comma-separated list of target platforms"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="outputs/campaign_output.md",
        help="Path to save the generated campaign brief"
    )
    args = parser.parse_args()

    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        print("Please configure it in a .env file or export it in your shell.")
        return

    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # 1. Load/Initialize Vector Store and retrieve context
    print("🔍 Initializing vector database and retrieving guidelines...")
    vector_db = load_vectorstore()
    rag_query = f"Marketing strategies for {args.brand} product {args.product} audience {args.audience}"
    brand_results = vector_db.similarity_search(rag_query, k=3)
    brand_context = "\n\n".join([doc.page_content for doc in brand_results])

    # 2. Configure Gemini LLM
    print("🤖 Configuring Gemini Large Language Model...")
    llm = LLM(
        model="gemini/gemma-4-31b-it",
        temperature=0.7,
        api_key=GEMINI_API_KEY,
        max_retries=6
    )

    platforms_list = [p.strip() for p in args.platforms.split(",")]

    # 3. Define Agents
    print("👥 Assembling the Marketing Crew (11 Agents)...")
    researcher = Agent(
        role="Trend Researcher",
        goal=f"Analyze current marketing trends, tactics, and strategies relevant to the campaign for brand {args.brand}",
        backstory="You are an expert market analyst who specializes in uncovering low-cost, high-impact growth channels for bootstrapped startups.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    competitor_analyst = Agent(
        role="Competitor Intelligence Analyst",
        goal=f"Analyze competitor positioning, marketing angles, strengths, and weaknesses for {args.brand}",
        backstory="You are a strategic intelligence specialist. You study competitor landing pages and messaging to identify gaps and differentiation opportunities.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    seo_specialist = Agent(
        role="SEO Specialist",
        goal=f"Construct a high-performance SEO strategy for {args.brand}'s product {args.product}",
        backstory="You are a data-driven SEO wizard. You excel at search intent analysis, keyword structuring, and building topical authority plans.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    content_strategist = Agent(
        role="Content Strategist",
        goal=f"Create a structured campaign strategy and content map for {args.brand}",
        backstory="You are a creative campaign planner. You synthesize raw trend data and competitor analysis into high-level content pillars and content calendars.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    writer = Agent(
        role="Content Writer",
        goal=f"Draft an engaging, authoritative, and direct blog article for brand {args.brand}. Adhere to writing voice guidelines.",
        backstory="You are an elite copywriter trained in direct-response writing. You avoid buzzwords at all costs and write punchy, conversational articles.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    social_manager = Agent(
        role="Social Media Manager",
        goal=f"Translate the written campaign assets into high-performance social posts for the selected platforms: {', '.join(platforms_list)}.",
        backstory="You are a social-native storyteller. You know how to hook attention on LinkedIn and structure Twitter threads.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    email_marketer = Agent(
        role="Email Marketer",
        goal=f"Draft direct-response email sequences and newsletters for {args.brand} that build trust and drive conversions.",
        backstory="You are an expert email marketer. You focus on personalization, clear subject lines, and compelling CTAs.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    campaign_planner = Agent(
        role="Campaign Planner",
        goal=f"Structure a detailed campaign launch timeline and platform selection blueprint for {args.brand} with budget {args.budget}.",
        backstory="You are an organized campaign coordinator. You determine where and when content should be distributed to maximize organic reach.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    performance_analyst = Agent(
        role="Performance Analytics Specialist",
        goal=f"Establish a complete measurement framework, KPIs, and tracking metrics for the {args.brand} campaign.",
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
        goal=f"Compile and format all inputs from the agents into a single, cohesive, premium marketing campaign report for {args.brand}.",
        backstory="You are a professional technical communicator. You structure complex strategy documents into readable executive summaries and markdown reports.",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # 4. Define Tasks
    print("📋 Defining Tasks...")
    research_task = Task(
        description=f"Perform detailed market research. Brand: {args.brand}, Product: {args.product}, Audience: {args.audience}, Goal: {args.goal}. Context: {brand_context}. Include: audience insights, current trends, and growth opportunities.",
        expected_output="Detailed marketing research report.",
        agent=researcher
    )

    competitor_task = Task(
        description=f"Analyze key competitors for {args.brand} in the {args.product} space. Identify their value propositions, marketing channels, and weaknesses.",
        expected_output="A competitive analysis report with 3 key differentiation opportunities.",
        agent=competitor_analyst
    )

    seo_task = Task(
        description=f"Develop a target SEO strategy for {args.brand}'s {args.product} targeting {args.audience}. Provide: 1 primary keyword, 3-5 LSI keywords, an SEO title, a meta description under 160 characters, and a H2/H3 outline.",
        expected_output="An SEO strategy blueprint containing target keywords and structured outline.",
        agent=seo_specialist
    )

    strategist_task = Task(
        description=f"Create a content strategy for {args.brand}. Define 3 primary content pillars and map them to customer awareness stages.",
        expected_output="A content strategy map detailing 3 content pillars, awareness stage alignments, and target messaging.",
        agent=content_strategist
    )

    writing_task = Task(
        description=f"Write a complete, high-quality, long-form blog article for {args.brand}'s {args.product} targeting {args.audience}. Follow the SEO and Content Strategy outlines.",
        expected_output="A complete, ready-to-publish blog article (700-1000 words) in a bold, conversational voice.",
        agent=writer
    )

    social_task = Task(
        description=f"Review the generated blog article and write social media promotions for platforms: {', '.join(platforms_list)}. Create 1 LinkedIn post using the 'hook -> story -> value -> CTA' structure and a Twitter/X thread of 4-6 tweets.",
        expected_output="A social media package including 1 LinkedIn post and 1 Twitter/X thread.",
        agent=social_manager
    )

    email_task = Task(
        description=f"Draft an email marketing package for {args.brand} consisting of 1 welcome email and a 2-part educational newsletter sequence on the topic of {args.product}.",
        expected_output="Complete email marketing copy sequence with subject lines, body copy, and CTAs.",
        agent=email_marketer
    )

    planning_task = Task(
        description=f"Develop a 4-week distribution calendar for {args.brand}'s {args.goal} campaign across platforms: {', '.join(platforms_list)}. Budget: {args.budget}.",
        expected_output="A structured 4-week campaign calendar and platform distribution schedule.",
        agent=campaign_planner
    )

    analytics_task = Task(
        description=f"Design a performance measurement framework for {args.brand}'s campaign. List specific metrics for Awareness, Engagement, Conversion, and Retention categories.",
        expected_output="A performance measurement matrix detailing KPIs, target benchmarks, and tracking methods.",
        agent=performance_analyst
    )

    compliance_task = Task(
        description=f"Audit all generated campaign assets for {args.brand} against brand guidelines, tone of voice rules, and compliance policies. Identify and flag any prohibited buzzwords.",
        expected_output="A brand compliance audit report highlighting issues found and corrective actions taken.",
        agent=brand_compliance_officer
    )

    reporting_task = Task(
        description=f"Synthesize the outputs of all 10 previous tasks for {args.brand} into a single comprehensive, executive-ready final marketing campaign report with a clear table of contents.",
        expected_output="A cohesive, end-to-end markdown report compiling the entire campaign package.",
        agent=executive_reporter
    )

    # 5. Assemble and run Crew
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

    print(f"\n🚀 Starting campaign generation for topic: '{args.topic}'...")
    
    async def run_crew():
        return await crew.kickoff_async(inputs={"topic": args.topic})

    result = asyncio.run(run_crew())

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(str(result))

    print(f"\n✅ Campaign successfully generated and saved to: {args.output}")

if __name__ == "__main__":
    main()
