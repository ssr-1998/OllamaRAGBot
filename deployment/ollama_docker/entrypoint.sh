#!/bin/sh
set -e

# Starting Ollama Server in the Background
ollama serve &

# Waiting to ensure Server is ready
sleep 3

# Pulling Model
ollama pull "$OLLAMA_MODEL"

# Now, keep the Server running
wait
