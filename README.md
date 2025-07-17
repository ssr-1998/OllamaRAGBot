# 🧠 Ollama_ChatBot — Context-Aware Local LLM ChatBot

A context-aware local ChatBot built with [Ollama](https://ollama.com/) that allows smooth multi-turn conversations, chat history preservation, and performance tracking. Powered by `Llama 3.2:1B`, this project bridges the gap between fast local inference and intelligent contextual memory — overcoming Ollama's default stateless nature.

> ✨ _Create domain-specific chat threads, track response times, and prepare for future fine-tuning or document-based discussions — all on your local machine._

---

## 🚀 Features

- ✅ **Context-Aware Chat**: Maintains prior conversation context to generate coherent responses.
- ✅ **Response Timer**: Measures and prints response time per prompt.
- ✅ **Persistent Chat Logging**: Saves each interaction in a structured JSON file (`Chat_History.json`).
- ✅ **Clean & Summarized Context Feeding**: Optimized context storage via preprocessing and summarization.
- 🔜 **Upcoming**:
  - [ ] Multiple chat threads under different **Chat Heads** (e.g., Data Science, Projects, Personal).
  - [ ] Fine-tuning models using curated history for better personalization.
  - [ ] Document-aware interaction (PDF, TXT, PY, Images).

---

## 🔧 Setup Instructions

### 1. ✅ Install Ollama
Download and install Ollama for your platform from:  
👉 https://ollama.com/

### 2. ⚙️ Pull & Run a Model

Choose a model based on your system resources. For this project:

```bash
ollama pull llama3.2
ollama run llama3.2
```

> 📌 Run `ollama` in terminal to verify installation.

### 3. 🐍 Setup Python Environment

```bash
# Clone this repo
git clone https://github.com/your-username/Ollama_ChatBot.git
cd Ollama_ChatBot

# Create a virtual environment (recommended)
python -m venv ollama-env
source ollama-env/bin/activate  # or `ollama-env\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### 4. ▶️ Run the ChatBot

```bash
python main_v2.py
```

Then start chatting! To exit, type `exit`.

---

## 📌 Example Interaction

```
You: What is LangChain?
Ollama: LangChain is an open-source framework for developing LLM applications...

 Response Time: 0 hours 0 minutes and 2.47 seconds.
```

---

## 🧠 Future Enhancements

- **Chat Heads**: Organize conversations by topic/domain.
- **Model Fine-Tuning**: Use past chat data to personalize and boost model performance.
- **Document-Aware Bot**: Allow uploading and querying PDFs, code files, or images.

---

## 📑 Requirements

- Python 3.8+
- Ollama installed and model pulled (e.g., `llama3.2`)
- Internet access only for initial model pull

---

## 🤝 Contributing

Suggestions or pull requests to improve functionality, add UIs, or document parsing are welcome!

---

## 📜 License

### MIT License

This project is licensed under the MIT License.

```
MIT License © 2025 Shubham Singh Rana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions...

(Full license text continues in LICENSE file)
```
