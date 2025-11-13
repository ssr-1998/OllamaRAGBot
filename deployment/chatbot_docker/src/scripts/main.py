from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from src.config import OLLAMA_API_URL, OLLAMA_MODEL
from src.utils import timer, maintain_chat_history, build_prompt_with_summary
from src.logger import get_logger
from src.exceptions import log_and_notify_exception

"""
Info:
Ollama Models Info: https://github.com/ollama/ollama
Model Used: Llama3.2:1B (1 Billion Parameters)
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
    model = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_API_URL)
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
