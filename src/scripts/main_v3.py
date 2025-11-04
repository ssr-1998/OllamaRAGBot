from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from src.utils import timer, maintain_chat_history, build_prompt_with_summary
from src.logger import get_logger
from src.exceptions import log_and_notify_exception

"""
Info:
Ollama Models Info: https://github.com/ollama/ollama
Model Used: Llama3.2:1B (1 Billion Parameters)

Plan:
    - Creating a ChromaDB Vector Database for storing & easy searching of the related sentences.
    - Converting Chats to Embeddings using Sentence Transformer.
    - Store in VectorDB.
    - On each new message, embed the latest input, query the DB for the most similar historical chats/conversation snippets.
    - Filter results for recency or relevance as required.
    - Use LangChain or OpenPrompt to assemble prompts using the retrieved context + the latest input.
    - Pass the constructed prompt to your Llama model via Ollama.
    - (For Independent Chats) Store session metadata (session_id) in your DB; design retrieval so only session-relevant embeddings are considered.

Steps:
    - pip install chromadb sentence-transformers
    - pip install transformers
    - pip install hf_xet (Installed to support the `t5-small` summarizer LLM model.)

Changes made:
    - Created Persistent ChromaDB Client
    - Added code for `Storing Human Message Embeddings into ChromaDB` in maintain_chat_history function.
    - Inserted 50 Chat History Entries to the `chat_history` Collection in ChromaDB.
    - Added a LLM Summarizer Layer to summarise Similar Messages fetched from ChromaDB.
"""

# ------------------- SETUP ----------------------

# Logging Setup
logger = get_logger(__name__)

# LLM Setup
template = """
You are a helpful Assistant knowledgeable about Previous Conversations. Be humble & greet the user.
Context is a generated summary from the historical line of questions I asked you on a similar topic.
Context: {context}
Now, answer this question: {question}
"""
model_loading_time = timer()
print("Loading LLM Model...")
try:
    model = OllamaLLM(model="llama3.2:1b")
except Exception as e:
    log_and_notify_exception(e, "Unable to load Ollama's LLM. Please try again later!")
    exit(1)
timer(model_loading_time)
prompt = ChatPromptTemplate.from_template(template)
_chain = prompt | model


def handle_conversation(_chain):

    print("\nWelcome to the AI ChatBot powered by Ollama Llama 3.2:1B Model! Type `exit` to quit.")

    while True:
        user_input = input("\nUser: ")

        if user_input.lower() == "exit":
            break

        else:
            print("\nThinking...")
            logger.info("New Query:\n")
            _context = build_prompt_with_summary(user_input)

            start_time = timer()

            print("Generating Response...")
            result = _chain.invoke({"context":_context, "question":user_input})

            timer(start_time)

            print("\nOllama: {}\n".format(result))

            maintain_chat_history(dict(human=user_input, context=_context, bot=result))  # Adding each Chat Entry in JSON


# ------------------- MAIN ------------------------

if __name__ == '__main__':
    handle_conversation(_chain)
