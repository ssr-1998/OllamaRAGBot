# Ollama's Base Image:
FROM ollama/ollama:latest

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Exposing the default Ollama Port
EXPOSE 11434

ENV OLLAMA_MODEL=llama3.2:1b

# Pull (Model - llama3.2:1b) on container startup & runserver:
# CMD ollama serve & sleep 3 && ollama pull llama3.2:1b && wait
ENTRYPOINT [ "/app/entrypoint.sh" ]
