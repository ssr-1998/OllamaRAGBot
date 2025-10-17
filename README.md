## 🧠 Project Summary
**OllamaRAGBot – A Local RAG-Powered Conversational System**

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for a local, privacy-friendly chatbot powered by **Ollama’s Llama 3.2:1B model**.  
The chatbot retrieves contextually relevant past conversations from a **ChromaDB vector database**, summarises them using a **lightweight transformer model**, and generates responses with fine-tuned contextual relevance.  
Built completely with open-source tools, it runs locally—no API keys or paid services required.

---

## 🧩 Ollama-RAG ChatBot (v3)

### Overview
**OllamaRAGBot** is an offline retrieval-based chatbot system that integrates **ChromaDB**, **Sentence Transformers**, and **local LLM inference** to maintain context-aware dialogues.  
The current version (`main_v3.py`) introduces:
- **RAG workflow** for dynamic context retrieval  
- **Context summarization layer** using `t5-small` for concise prompting  
- **Query filtration + lexical deduplication** for high topic relevance  

---

## 🧱 Project Structure

📁 Ollama<br>
├── 📄 .gitignore<br>
├── 📄 LICENSE<br>
├── 📄 README.md<br>
├── 📁 data<br>
│ ├── 📄 Chat_History.json # Auto-created if not present<br>
│ ├── 📁 ChromaDB_Storage/ # Auto-created persistent vector DB<br>
│ └── 📄 Sample_Chat_History.json # For GitHub Reference<br>
├── 📄 requirements.txt<br>
├── 📁 src<br>
│ ├── 📄 init.py<br>
│ ├── 📁 scripts<br>
│ │ ├── 📄 init.py<br>
│ │ ├── 📄 main_v1.py<br>
│ │ ├── 📄 main_v2.py<br>
│ │ └── 📄 main_v3.py # Latest version with RAG pipeline<br>
└── 📁 tests<br>
├── 📄 chromadb_test.py<br>
└── 📄 test.py<br>

> **Note:**  
> - If `data/Chat_History.json` or `data/ChromaDB_Storage/` folder does not exist, they are automatically generated when the chatbot runs.  
> - `data/ChromaDB_Storage/` and `data/Chat_History.json` are ignored in `.gitignore` to prevent unnecessary repository bloat.

---

## 🧠 Key Features
- **Retrieval-Augmented Generation (RAG)** with persistent vector database
- **Lightweight summarization** to reduce context load
- **Lexical filtration** for topic-specific retrieval
- **Deduplication layer** to improve summarization precision
- **Automatic context memory expansion** across sessions
- **Fully offline** and local execution with open-source libraries

---

## ⚙️ Technologies
- **Python 3.8+**
- **Ollama (Llama 3.2:1B) or the model of your choice (https://ollama.com/library)**
- **ChromaDB** for vector storage
- **SentenceTransformers** for text embeddings
- **Hugging Face Transformers (T5-small)** for summarization
- **LangChain Core** for chaining prompts and model inference

---

## 🔧 Setup Instructions

### 1. ✅ Install Ollama

Download and install Ollama for your platform from:  
👉 https://ollama.com/

### 2. ⚙️ Pull & Run a Model

Choose a model based on your system resources. For this project:

```bash
ollama pull llama3.2:1b  # Pull a Model

ollama run llama3.2:1b  # Run a Model
```

> 📌 Run `ollama` in terminal to verify installation.

### Clone the Repository

```bash
git clone https://github.com/your-username/OllamaRAGBot.git

cd OllamaRAGBot
```

### Create a virtual environment (Recommended)

```bash
python -m venv ollama-env

source ollama-env/bin/activate  # or `ollama-env\Scripts\activate` on Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Chatbot

```bash
python src/scripts/main_v3.py
```

> Then start chatting! To exit, type `exit`.
> The chatbot automatically creates missing data storage or JSON history files on first run.

---

## 📌 Example Interaction

```
You: What is LangChain?
Ollama: LangChain is an open-source framework for developing LLM applications...

 Response Time: 0 hours 0 minutes and 1.47 seconds.
```

---

## 🧩 Version History
- **v1:** Basic question–answer conversational template  
- **v2:** Integrated contextual chaining with LangChain prompt templates  
- **v3 (Current):** RAG pipeline with filtration, summarisation, and deduplication  

---

## 🧭 Future Plans
1. Adding Logging & Exception Handling  
2. Modularising chatbot components into sub-pipelines  
3. Fine-tuning the LLM Model  

---

## 🤝 Contributing
Contributions welcome!  
If you find improvements or have ideas for new features (like caching, feedback scoring, or knowledge graph integration), feel free to fork and pull a request.

---

## 📄 License
This project is distributed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.
