import os, re, json, logging, chromadb, pathlib
from nltk.corpus import stopwords
from langchain_ollama import OllamaLLM
from transformers import pipeline
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

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

# Silencing INFO and WARNING logs for the transformers but keep ERRORS visible.
logging.getLogger("transformers").setLevel(logging.ERROR)

# Loading English Stopwords for the Filteration Layer of the Relevant Context Generation
stop_words = set(stopwords.words("english"))

# Embedding & Summarization Models
embedder = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="t5-small")

# Persistent Vector DB - ChromaDB
persistentDB_path = pathlib.Path("data/ChromaDB_Storage")
persistentDB_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist.
client = chromadb.PersistentClient(path="data/ChromaDB_Storage")
collection = client.get_or_create_collection(name="chat_history")

# LLM Setup
template = """
You are a helpful Assistant knowledgeable about Previous Conversations. Be humble & greet the user.
Context is a generated summary from the historical line of questions I asked you on a similar topic.
Context: {context}
Now, answer this question: {question}
"""

model = OllamaLLM(model="llama3.2:1b")
prompt = ChatPromptTemplate.from_template(template)
_chain = prompt | model

# ------------------- FUNCTIONS -------------------


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
    """
    Registers each Chat Entry into a JSON File named `Chat_History.json` for data collection and Stores Human Message as Embeddings in ChromaDB.

    :param chat_entry: a dictionary that consists Question, Context, & Answer from the Bot.
    :return: None
    """
    json_file_path = "data/Chat_History.json"

    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            data = json.load(file)

    else:
        data = dict(chat_history=list())

    data["chat_history"].append(chat_entry)

    chat_idx = len(data["chat_history"]) - 1

    with open(json_file_path, "w") as f:
        json.dump(data, f, indent=4)
    
    # Storing Human Message Embeddings into ChromaDB
    human_text = chat_entry["human"]
    embedding = embedder.encode([human_text]).tolist()

    chroma_id = "chat_{}".format(str(chat_idx))

    collection.add(
        documents=[human_text], 
        embeddings=embedding, 
        metadatas=[{"index": chat_idx}], 
        ids=[chroma_id]
    )

    print("\nChat History Updated!\n")

    return None


def clean_and_tokenize(text: str):

    clean_text = re.sub(r"\W+", " ", text.lower())

    tokens = [t for t in clean_text.split() if t not in stop_words]

    return set(tokens)


def deduplicate_queries(filtered_queries, tokenized_dict, threshold=0.7):
    """
    Deduplicate filtered queries by comparing their meaning via token overlap (Jaccard similarity).
    """
    deduped = []
    seen_tokens = []

    for i, q in enumerate(filtered_queries):
        q_tokens = tokenized_dict.get(q)
        is_duplicate = False

        for j, existing_tokens in enumerate(seen_tokens):
            # Jaccard similarity
            intersection = len(q_tokens & existing_tokens)
            union = len(q_tokens | existing_tokens)
            score = intersection / union if union else 0.0

            if score >= threshold:
                is_duplicate = True
                print(f"Duplicate check: '{q}' vs '{deduped[j]}' - Match score: {score:.2f}")
                break

        if not is_duplicate:
            deduped.append(q)
            seen_tokens.append(q_tokens)

    return deduped


def filter_relevant_queries(user_input, relevant_queries, threshold=0.3):

    # Part-1: Checking whether the relevant queries we got from the ChromaDB are actually relevant to the User Input or not.
    user_tokens = clean_and_tokenize(user_input)

    filtered_1 = []
    relevant_query_tokens = dict()

    for q in relevant_queries:
        q_tokens = clean_and_tokenize(q)

        if user_tokens and q_tokens:
            overlap = len(user_tokens & q_tokens) / max(len(user_tokens), 1)

            if overlap >= threshold:
                filtered_1.append(q)
                relevant_query_tokens[q] = q_tokens

    print(f"\nSelected {len(filtered_1)} relevant queries out of {len(relevant_queries)} after filtration part-1.\n")

    # Part-2: De-duplication check for the filtered relevant queries. Such that, no repetitive queries are used for context building.
    filtered_2 = deduplicate_queries(filtered_1, relevant_query_tokens)
    if len(filtered_1) != len(filtered_2):
        print(f"\nSelected {len(filtered_2)} relevant queries out of {len(filtered_1)} after filtration part-2.")

    return filtered_2


def summarize_context(context_list):

    text = " ".join(context_list)

    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    return summary


def build_prompt_with_summary(user_input, n_results=10):

    results = collection.query(
        query_embeddings=embedder.encode([user_input]).tolist(),
        n_results=n_results
    )
    
    raw_contexts = results['documents'][0] if results['documents'] else []

    if len(raw_contexts) > 0:
        raw_contexts = filter_relevant_queries(user_input=user_input, relevant_queries=raw_contexts)

    summarized_context = summarize_context(raw_contexts) if raw_contexts else ""

    return summarized_context


def handle_conversation(_chain):

    print("Welcome to the AI ChatBot powered by Ollama Llama 3.2:1B Model! Type `exit` to quit.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        else:
            _context = build_prompt_with_summary(user_input)

            result = _chain.invoke({"context":_context, "question":user_input})

            print("Ollama: {}\n".format(result))

            maintain_chat_history(dict(human=user_input, context=_context, bot=result))  # Adding each Chat Entry in JSON


# ------------------- MAIN ------------------------

if __name__ == '__main__':
    handle_conversation(_chain)
