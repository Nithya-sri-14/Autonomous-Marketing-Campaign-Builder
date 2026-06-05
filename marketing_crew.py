import os
import argparse
from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv()

from crewai import Agent, Task, Crew, LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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
    parser = argparse.ArgumentParser(description="Autonomous Marketing Campaign Builder (CLI Version)")
    parser.add_argument(
        "--topic", 
        type=str, 
        default="AI-driven organic marketing for bootstrapped startups",
        help="Campaign topic for the multi-agent system"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="outputs/campaign_output.md",
        help="Path to save the generated campaign brief"
    )
    args = parser.parse_args()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("❌ Error: GEMINI_API_KEY environment variable is not set.")
        print("Please configure it in a .env file or export it in your shell.")
        return

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # 1. Load/Initialize Vector Store and retrieve context
    print("🔍 Initializing vector database and retrieving guidelines...")
    vector_db = load_vectorstore()
    brand_results = vector_db.similarity_search("startup writing tone tone and style", k=2)
    brand_context = "\n\n".join([doc.page_content for doc in brand_results])

    print("📚 Retrieved Brand Voice Guidelines:")
    print("-" * 60)
    print(brand_context)
    print("-" * 60)

    # 2. Configure Gemini LLM
    print("🤖 Configuring Gemini Large Language Model...")
    llm = LLM(
        model="gemini/gemini-2.5-flash",
        temperature=0.7,
        api_key=GEMINI_API_KEY,
        max_retries=6
    )

    # 3. Define Agents
    print("👥 Assembling the Marketing Crew...")
    researcher = Agent(
        role="Trend Researcher",
        goal="Analyze current marketing trends, tactics, and strategies relevant to the campaign topic: {topic}",
        backstory="You are an expert market analyst who specializes in uncovering low-cost, high-impact growth channels for bootstrapped startups. You analyze what works now, filtering out corporate fluff in favor of actual actionable traction.",
        verbose=True,
        llm=llm
    )

    seo_specialist = Agent(
        role="SEO Specialist",
        goal="Construct a high-performance SEO strategy around the campaign topic: {topic}",
        backstory="You are a data-driven SEO wizard. You excel at search intent analysis, keyword structuring, schema planning, and building topical authority plans that require $0 in ad spend.",
        verbose=True,
        llm=llm
    )

    writer = Agent(
        role="Content Writer",
        goal="Draft an engaging, authoritative, and direct blog article on: {topic}. You MUST inject the exact brand voice guidelines provided in the context.",
        backstory="You are an elite copywriter trained in direct-response writing. You avoid buzzwords like 'delve' or 'tapestry' at all costs. You write punchy, conversational, and incredibly high-value articles that founders love reading.",
        verbose=True,
        llm=llm
    )

    social_manager = Agent(
        role="Social Media Manager",
        goal="Translate the written campaign assets into high-performance social posts for LinkedIn and a Twitter/X thread.",
        backstory="You are a social-native storyteller. You know exactly how to hook attention on LinkedIn with value-packed posts and how to structure engaging Twitter/X threads that get bookmarked and reshared.",
        verbose=True,
        llm=llm
    )

    # 4. Define Tasks
    research_task = Task(
        description=(
            "Conduct thorough research on '{topic}'. Identify 3 high-impact organic channels, "
            "current trends, and specific growth tactics that bootstrapped startups are using. "
            "Focus on actionable case studies or direct strategies."
        ),
        expected_output=(
            "A structured research brief summarizing key organic marketing trends, "
            "top 3 acquisition channels, and 3 case-study-style growth tactics."
        ),
        agent=researcher
    )

    seo_task = Task(
        description=(
            "Based on the research brief, develop a target SEO strategy for '{topic}'. "
            "Provide: 1 primary keyword, 3-5 LSI/secondary keywords, an engaging SEO Title, "
            "a meta description under 160 characters, and a recommended article outline (H2/H3 headers)."
        ),
        expected_output=(
            "An SEO strategy blueprint containing: target keywords, optimized meta-tags, "
            "and a structured H2/H3 article outline."
        ),
        agent=seo_specialist
    )

    writing_task = Task(
        description=(
            "Write a complete, high-quality, long-form blog article on '{topic}'.\n\n"
            "IMPORTANT RULES:\n"
            "1. You MUST follow the SEO outline and incorporate target keywords from the SEO task.\n"
            "2. You MUST strictly adhere to the brand guidelines provided below. Do NOT use any prohibited buzzwords.\n\n"
            "BRAND GUIDELINES CONTEXT:\n"
            f"{brand_context}"
        ),
        expected_output=(
            "A complete, ready-to-publish blog article (700-1000 words) written in a bold, "
            "conversational, founder-focused startup voice with proper headers, short paragraphs, "
            "and zero prohibited buzzwords."
        ),
        agent=writer
    )

    social_task = Task(
        description=(
            "Review the generated blog article and write social media promotions:\n"
            "1. One long-form LinkedIn post that uses the 'hook -> story -> value -> CTA' structure.\n"
            "2. An engaging Twitter/X thread (4-6 tweets) summarizing the article's core points, "
            "where each tweet is conversational, punchy, and fits within character limits."
        ),
        expected_output=(
            "A social media package including 1 value-packed LinkedIn post and 1 engaging "
            "Twitter/X thread (4-6 tweets separated by '---' or 'Tweet X/X:')."
        ),
        agent=social_manager
    )

    # 5. Create Crew and run execution
    marketing_crew = Crew(
        agents=[researcher, seo_specialist, writer, social_manager],
        tasks=[research_task, seo_task, writing_task, social_task],
        verbose=True
    )

    print(f"\n🚀 Starting campaign generation for topic: '{args.topic}'...")
    campaign_result = marketing_crew.kickoff(inputs={"topic": args.topic})

    # Save outputs
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(str(campaign_result))

    print(f"\n✅ Campaign successfully generated and saved to: {args.output}")

if __name__ == "__main__":
    main()
