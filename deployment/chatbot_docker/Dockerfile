# Python's Base Image:
FROM python:3.11.9-slim

# Working Directory:
WORKDIR /app

# Copying Requirements & Source Code:
COPY requirements_linux.txt .
COPY src ./src
COPY src/scripts ./src/scripts

# Installing required dependencies:
RUN pip install --no-cache-dir -r requirements_linux.txt

# Downloading English Stopwords from NLTK Library:
RUN python -m nltk.downloader stopwords

# Environment Variable for Ollama endpoint:
ENV OLLAMA_API_URL=http://ollama-server:11434
ENV OLLAMA_MODEL=llama3.2:1b

# Default Run Command:
CMD [ "python", "-m", "src.scripts.main" ]
