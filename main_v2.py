import json
import os
import re
from nltk.corpus import stopwords
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

"""
Info:
Ollama Models Info: https://github.com/ollama/ollama
Model Used: Llama3.2:1B (1 Billion Parameters)

Changes made:
    - Created & Applied the `timer` Function.
    - Defined `extract_content_for_context` Function.
    - Modified `handle_conversation` Function as per `modify_context` Function.
    - Defined & Applied the `clean_the_text` Function.
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


def clean_the_text(text:str):
    """
    Function that applies a specific set of Text Cleaning Rules and returns Clean Concise Text.
    :param text: Any String
    :return: A Cleaned String
    """
    _stopwords = list(stopwords.words("english")) + ["hi", "hello", "bot", "buddy", "user:", "ollama:", "human:"]
    text = text.lower()  # Lower Casing each word
    text = " ".join(t for t in text.split() if t not in _stopwords)  # Removing the StopWords

    # Extracting and Storing aside the Alpha Numeric items from the text:
    match = re.finditer("\\s(\\w+[.:-_]){2,5}\\w+\\s", text)
    match_lst = list(match)
    if len(match_lst) > 0:
        lst = []
        for i in match_lst:
            lst.append(i.group())
        alphaNumerics = " ".join(i for i in lst)

    text = re.sub('((www.[^s]+)|(https?://[^s]+))', ' ', text)  # Cleaning URLs
    text = re.sub("\\S*\\d\\S*", "", text).strip()  # Cleans any number, alphanumerics, etc
    text = re.sub('[^A-Za-z0-9 ]+', " ", text)  # Cleans any special characters

    # Adding Alpha Numerics back to the text.
    if len(match_lst) > 0:
        text += alphaNumerics

    # Removing Extra Spaces:
    text = re.sub("^ ", "", text)
    text = re.sub(" $", "", text)
    text = re.sub(" +", " ", text)
    return text


def extract_content_for_context(question:str, answer:str):
    """
    A function that takes the Current Sessions Question & Answer as input, applies a text cleaning function and
    combines them to add data to Context for the Model.
    :param question: The Question asked from the Bot.
    :param answer: Answer received from the Bot.
    :return: A combined & summarized version of the contents of the Question & the Answer.
    """
    final_content = "\nQUESTION: {}. ANSWER: {}\n".format(clean_the_text(question), clean_the_text(answer))
    return final_content


def maintain_chat_history(chat_entry :dict):
    """
    Registers each Chat Entry into a JSON File named `Chat_History.json` for data collection.
    :param chat_entry: a dictionary that consists Question, Context, & Answer from the Bot.
    :return: None
    """
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
            context += extract_content_for_context(user_input, result)


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

# ToDo: Start Keyword based actions for the Ollama.
