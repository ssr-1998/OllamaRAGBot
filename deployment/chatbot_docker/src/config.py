import os

# Setting Ollama API base URL (Override this if Ollama is running as a seperate Docker Container)
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
