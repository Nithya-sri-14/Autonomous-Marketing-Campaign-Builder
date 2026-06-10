# 🚀 Autonomous Marketing Campaign Builder with CrewAI + Gemini + RAG

Welcome to the **Autonomous Marketing Campaign Builder**, a production-ready, multi-agent digital marketing automation platform. This system utilizes a **Multi-Agent Orchestration Framework (CrewAI)** combined with a **Google Gemini Large Language Model**, local **HuggingFace Embeddings**, and a **ChromaDB Vector Database for Retrieval-Augmented Generation (RAG)** to produce high-value, organic marketing campaigns matching a specific brand voice.

It supports both an interactive **Streamlit Web Application** for a visual experience and a **Command Line Interface (CLI)** script for programmatic, automated runs. Additionally, it integrates directly with **n8n** automation workflows.

---

## 📐 Multi-Agent Workflow & RAG Architecture

The workflow progresses dynamically across specialized stages:

```text
[User Campaign Topic Input]
           │
           ▼
[ChromaDB Vector Search] ──► (Retrieves brand writing voice, prohibited words) ──┐
                                                                                ▼
[CrewAI Orchestrator Engine] ◄───────────────────────────────────── [Google Gemini]
           │
           ├──► 🧑‍💻 Agent 1: Trend Researcher
           │           │   └─ Goal: Analyze current marketing channels & growth tactics.
           │           ▼
           ├──► 📊 Agent 2: Competitor Intelligence Analyst
           │           │   └─ Goal: Analyze competitor positioning, strengths, and weaknesses.
           │           ▼
           ├──► 🔍 Agent 3: SEO Specialist
           │           │   └─ Goal: Generate keywords, optimize page title, structure, & meta tags.
           │           ▼
           ├──► 📝 Agent 4: Content Strategist
           │           │   └─ Goal: Create structured campaign strategy and content map.
           │           ▼
           ├──► ✍️ Agent 5: Content Writer  ◄── [RAG Context Injected (data/brand.txt guidelines)]
           │           │   └─ Goal: Draft high-value, direct blog articles without buzzwords.
           │           ▼
           ├──► 📱 Agent 6: Social Media Manager
           │           │   └─ Goal: Translate campaign into punchy LinkedIn posts & Twitter threads.
           │           ▼
           ├──► 📧 Agent 7: Email Marketer
           │           │   └─ Goal: Draft direct-response email sequences and newsletters.
           │           ▼
           ├──► 🗓️ Agent 8: Campaign Planner
           │           │   └─ Goal: Structure campaign launch timeline, platform selection, & budget.
           │           ▼
           ├──► 📈 Agent 9: Performance Analytics Specialist
           │           │   └─ Goal: Establish measurement framework, KPIs, and tracking metrics.
           │           ▼
           ├──► 👮 Agent 10: Brand Compliance Officer
           │           │   └─ Goal: Audit assets for brand consistency, compliance, & prohibited buzzwords.
           │           ▼
           └──► 🎤 Agent 11: Executive Reporter
                       │   └─ Goal: Compile all inputs into a single cohesive, premium marketing campaign report.
                       ▼
             [Final Markdown Campaign Package]
             ├─ Research Brief
             ├─ Competitive Analysis
             ├─ SEO Keyword Blueprint
             ├─ Content Strategy Map
             ├─ Complete Long-form Blog Post
             ├─ Value-driven Social Content (LinkedIn + Twitter thread)
             ├─ Email Marketing Package
             ├─ Campaign Launch Plan
             ├─ Performance Measurement Matrix
             ├─ Brand Compliance Audit
             └─ Executive Campaign Report
```

---

## 🛠️ Project Structure

```text
├── data/
│   └── brand.txt             # Brand identity, tone guidelines, and prohibited words (RAG source)
├── outputs/                  # Directory where CLI campaign output markdown briefs are saved
├── app.py                    # Streamlit Web Application (Includes custom premium UI & n8n triggers)
├── marketing_crew.py         # CLI execution script running the 11-agent campaign builder
├── n8n_workflow.json         # Ready-to-import n8n workflow for post-generation distribution
├── requirements.txt          # Python dependencies
├── .env.example              # Sample environment configuration file
├── .gitignore                # Git files/directories to ignore (keeps API keys and databases local)
├── Autonomous_Marketing_Campaign_Builder.ipynb # Google Colab Notebook
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9** or higher installed on your system.
- A **Google Gemini API Key** (Get one from [Google AI Studio](https://aistudio.google.com/)).

### 2. Installation

Clone this repository or extract it to a directory, then open your terminal inside the project directory:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS / Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install the dependencies
pip install -r requirements.txt
```

> [!NOTE]
> On Linux servers or Google Colab environments, `pysqlite3-binary` will automatically patch the standard SQLite library for ChromaDB compatibility. On macOS and Windows, standard sqlite3 works out of the box.

### 3. Environment Configuration

Copy the sample environment file to create your own configuration:

```bash
cp .env.example .env
```

Open the `.env` file and insert your Google Gemini API key:

```env
GEMINI_API_KEY=AIzaSy...your_gemini_api_key...
```

---

## 🖥️ Running the Application

This project offers two ways to run the marketing campaign builder.

### Option A: Streamlit Web UI (Visual App)

Launch the interactive web application to customize campaign goals, target audience, budget, platforms, and preview outputs in real-time:

```bash
streamlit run app.py
```

Open the URL provided in your terminal (usually `http://localhost:8501`).

- **Auto RAG Database Build**: The Streamlit app checks if `./chroma_db` is populated. If not, it automatically reads your `data/brand.txt` file, chunks the text, creates embeddings using local sentence-transformer models, and builds the Chroma database.
- **Custom CSS Theme**: The application features a premium dark-glassmorphism theme with modern typography ("Plus Jakarta Sans").
- **n8n Webhook Trigger**: Easily input your n8n production webhook URL in the UI to push generated marketing campaigns directly into automated delivery pipelines.

### Option B: CLI Marketing Crew (Terminal Script)

Run the multi-agent system directly from your command line to automate campaign generation.

```bash
# Run with default settings
python marketing_crew.py

# Run with custom campaign settings
python marketing_crew.py --topic "SEO tactics for local SaaS tools" --brand "MySaaS" --product "SEO Generator" --audience "SaaS founders" --output "outputs/saas_campaign.md"
```

---

## 🔌 n8n Workflow Automation

We've bundled an automated delivery workflow in **`n8n_workflow.json`**. 

This workflow triggers when the web application posts the campaign report payload. It:
1. Receives the webhook trigger.
2. Parses and separates the campaign report sections (Research, Blog post, Social content).
3. Automatically drafts and sends an email notification with the consolidated package.
4. Alerts the marketing channels via Telegram, staging the LinkedIn post and Instagram caption draft.

### Setting Up n8n:
1. Open your **n8n** instance.
2. Click on **Workflows** -> **Import from file...** and select `n8n_workflow.json`.
3. Configure your Gmail and Telegram credentials in the corresponding nodes.
4. Save and **Activate** the workflow.
5. Copy the production **Webhook URL** and paste it into the Streamlit Web Application when running.

---

## ☁️ AWS Elastic Beanstalk Deployment

This project is fully compatible with **AWS Elastic Beanstalk (Python Platform)**.

### 1. Platform Details
- **Platform**: Python
- **Platform Branch**: `Python 3.11 running on 64bit Amazon Linux 2023` (or `Python 3.10`)

### 2. AWS Compatibility Measures
- **Streamlit Port Routing**: The bundle includes a `Procfile` at the root which configures Streamlit to run on port `8000` (which AWS Elastic Beanstalk uses to proxy HTTP traffic).
- **SQLite3 Patch**: AWS runs Amazon Linux which comes with an older version of SQLite. The code contains an automatic patch (`pysqlite3-binary`) that overrides the system sqlite3 dynamically on start.
- **Cache Directory**: We configure the HuggingFace cache directory (`HF_HOME`) to `/tmp/huggingface` in the code, ensuring the models can be downloaded to a writable location on EC2 instances.

### 3. Deployment Steps
1. **Zip the Project**:
   Compress your project files into a `.zip` archive. Make sure you **do not** include `venv/` or any local `chroma_db/` folder in the zip file.
   ```bash
   zip -r deploy.zip . -x "venv/*" "chroma_db/*" ".git/*" "*.pyc"
   ```
2. **Create Beanstalk Application & Environment**:
   - Go to the AWS Elastic Beanstalk console and click **Create Application**.
   - Set the Platform to **Python**.
   - Under **Application code**, select **Upload your code** and upload your `deploy.zip` file.
3. **Configure Environment Variables**:
   - In your Beanstalk Environment configuration, go to **Configuration** -> **Updates, monitoring, and logging** -> **Platform properties** (Environment properties).
   - Add an environment property:
     - Name: `GEMINI_API_KEY`
     - Value: `your_gemini_api_key`
   - Save and apply the configuration.
4. **Access the App**:
   Once the environment health shows **Green**, click the environment URL to open your Streamlit app!
