import json
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

"""
Info:
Ollama Models Info: https://github.com/ollama/ollama
Model Used: Llama3.2:1B (1 Billion Parameters)
"""


def timer(start_time=None):
    """Function to calculate total time taken:"""
    from datetime import datetime
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Response Time: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def maintain_chat_history(chat_entry :dict):
    json_file_path = "Chat_History.json"

    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            data = json.load(file)
    else:
        data = dict(chat_history=list())

    data["chat_history"].append(chat_entry)

    with open(json_file_path, "w") as f:
        json.dump(data, f, indent=4)


def handle_conversation(_chain):
    context = ""
    print("Welcome to the AI ChatBot powered by Ollama Llama 3.2:1B Model! Type `exit` to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        else:
            start_time = timer()
            result = _chain.invoke({"context":context, "question":user_input})
            timer(start_time)
            print("Ollama: {}\n".format(result))
            maintain_chat_history(dict(human=user_input, context=context, bot=result))  # Adding each Chat Entry in JSON
            context += f"\nUser: {user_input}\nOllama: {result}"


if __name__ == '__main__':
    template = """
    Answer the question below:
    Here is the Conversation history: {context}
    Question: {question}
    Answer: 
    """  # It's just a sample template that we give to the Model to understand that it has to work in Question-Answer Format.

    model = OllamaLLM(model="llama3.2:1b")
    prompt = ChatPromptTemplate.from_template(template)
    """
    Next operation will combine the above two operations in a chain such that first the Prompt will be collected
    from the Template Var which will have the Context & Question, then it will go to the Model.
    """
    _chain = prompt | model

    handle_conversation(_chain)
