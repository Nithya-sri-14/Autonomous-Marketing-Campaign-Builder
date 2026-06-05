# 🚀 Autonomous Marketing Campaign Builder with CrewAI + RAG + Gemini

Welcome to the **Autonomous Marketing Campaign Builder**, a production-ready, multi-agent digital marketing automation platform. This system utilizes a **Multi-Agent Orchestration Framework (CrewAI)** combined with a **Google Gemini Large Language Model**, local **HuggingFace Embeddings**, and a **ChromaDB Vector Database for Retrieval-Augmented Generation (RAG)** to produce high-value, organic marketing campaigns matching a specific brand voice.

It supports both an interactive **Streamlit Web Application** for a visual experience and a **Command Line Interface (CLI)** script for programmatic, automated runs. Additionally, it integrates directly with **n8n** automation workflows.

---

## 📐 Multi-Agent Workflow & RAG Architecture

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
           ├──► 🔍 Agent 2: SEO Specialist
           │           │   └─ Goal: Generate keywords, optimize page title, structure, & meta tags.
           │           ▼
           ├──► ✍️ Agent 3: Content Writer  ◄── [RAG Context Injected (data/brand.txt guidelines)]
           │           │   └─ Goal: Draft high-value, direct blog articles without buzzwords.
           │           ▼
           └──► 📱 Agent 4: Social Media Manager
                       │   └─ Goal: Translate campaign into punchy LinkedIn posts & Twitter threads.
                       ▼
             [Final Markdown Campaign Package]
             ├─ Research Brief
             ├─ SEO Keyword Blueprint
             ├─ Complete Long-form Blog Post
             └─ Value-driven Social Content (LinkedIn + Twitter thread)
```

---

## 🛠️ Project Structure

```text
├── data/
│   └── brand.txt             # Brand identity, tone guidelines, and prohibited words (RAG source)
├── outputs/                  # Directory where CLI campaign output markdown briefs are saved
├── app.py                    # Streamlit Web Application (Includes custom premium UI)
├── marketing_crew.py         # CLI execution script running the 4-agent campaign builder
├── requirements.txt          # Python dependencies
├── .env.example              # Sample environment configuration file
└── .gitignore                # Git files/directories to ignore (keeps API keys and databases local)
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

- **Auto RAG Database Build**: The Streamlit app checks if `./chroma_db` is populated. If not, it automatically reads your `data/brand.txt` file, chunks the text, creates embeddings using local HuggingFace models, and builds the Chroma database.
- **Custom CSS Theme**: The application features a premium dark-glassmorphism theme with modern typography ("Plus Jakarta Sans").

### Option B: CLI Marketing Crew (Terminal Script)

Run the multi-agent system directly from your command line to automate campaign generation.

```bash
# Run with default topic
python marketing_crew.py

# Run with custom campaign topic
python marketing_crew.py --topic "Newsletter growth strategies for local retail shops" --output "outputs/newsletter_campaign.md"
```

---

## ☁️ AWS Elastic Beanstalk Deployment

This project is fully compatible with **AWS Elastic Beanstalk (Python Platform)**.

### 1. Platform Details
- **Platform**: Python
- **Platform Branch**: `Python 3.11 running on 64bit Amazon Linux 2023` (or `Python 3.10`)

### 2. AWS Compatibility Measures
- **Streamlit Port Routing**: The bundle includes a `Procfile` at the root which configures Streamlit to run on port `5000` (which AWS Elastic Beanstalk uses to proxy HTTP traffic).
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

---

## 📤 Pushing to GitHub

To store this code on your GitHub account, run these commands in your project root folder:

```bash
# Initialize git repository
git init

# Add all files to staging (sensitive credentials are ignored automatically via .gitignore)
git add .

# Create the initial commit
git commit -m "Initial commit: Autonomous Marketing Campaign Builder"

# Create a new repository on github.com (without README or gitignore), then link it:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git

# Push the code to GitHub
git push -u origin main
```
