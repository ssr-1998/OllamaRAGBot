import os, re, json, pathlib, chromadb
from nltk.corpus import stopwords
from transformers import pipeline
from src.logger import get_logger
from src.exceptions import log_and_notify_exception
from sentence_transformers import SentenceTransformer

# ------------------- SETUP ----------------------

# Logging Setup
logger = get_logger(__name__)

# Loading English Stopwords for the Filteration Layer of the Relevant Context Generation
try:
    stop_words = set(stopwords.words("english"))
except Exception as e:
    log_and_notify_exception(e, "Requirements unsatisfied. Please download english stopwords from nltk library.")
    exit(1)

# Embedding & Summarization Models
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="t5-small")
except Exception as e:
    log_and_notify_exception(e, "Unable to load Embedding/Summarization Model. Please try again later!")
    exit(1)

# Persistent Vector DB - ChromaDB
print("Loading ChromaDB...")
try:
    persistentDB_path = pathlib.Path("data/ChromaDB_Storage")
    persistentDB_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist.
    client = chromadb.PersistentClient(path="data/ChromaDB_Storage")
    collection = client.get_or_create_collection(name="chat_history")
except Exception as e:
    log_and_notify_exception(e, "Encountered into a DB problem. Please check logs for details and try again later.")
    exit(1)

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
        logger.info('Response Time: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def maintain_chat_history(chat_entry :dict):
    """
    Registers each Chat Entry into a JSON File named `Chat_History.json` for data collection and Stores Human Message as Embeddings in ChromaDB.

    :param chat_entry: a dictionary that consists Question, Context, & Answer from the Bot.
    :return: None
    """
    try:
        logger.info("Maintaining Chat History...")
        json_file_path = os.path.join(os.getcwd(), "data", "Chat_History.json")

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
        embedding = embedder.encode([human_text], show_progress_bar=False).tolist()

        chroma_id = "chat_{}".format(str(chat_idx))

        collection.add(
            documents=[human_text], 
            embeddings=embedding, 
            metadatas=[{"index": chat_idx}], 
            ids=[chroma_id]
        )

        logger.info("Chat History Updated!")
    except Exception as e:
        log_and_notify_exception(e, "Unable to update Chat History. Please check the logs for details.")

    return None


def clean_and_tokenize(text: str):

    clean_text = re.sub(r"\W+", " ", text.lower())

    tokens = [t for t in clean_text.split() if t not in stop_words]

    return set(tokens)


def deduplicate_queries(filtered_queries, tokenized_dict, threshold=0.7):
    """
    Deduplicate filtered queries by comparing their meaning via token overlap (Jaccard similarity).
    """
    try:
        deduped = []
        seen_tokens = []

        for q in filtered_queries:
            q_tokens = tokenized_dict.get(q)
            is_duplicate = False

            for j, existing_tokens in enumerate(seen_tokens):
                # Jaccard similarity
                intersection = len(q_tokens & existing_tokens)
                union = len(q_tokens | existing_tokens)
                score = intersection / union if union else 0.0

                if score >= threshold:
                    is_duplicate = True
                    logger.info(f"Duplicate check: '{q}' vs '{deduped[j]}' - Match score: {score:.2f}")
                    break

            if not is_duplicate:
                deduped.append(q)
                seen_tokens.append(q_tokens)
    except Exception as e:
        log_and_notify_exception(e, "Unable to check De-duplication of the related historical chat queries. Please check logs for more details.")
        deduped = filtered_queries

    return deduped


def filter_relevant_queries(user_input, relevant_queries, threshold=0.3):

    # Part-1: Checking whether the relevant queries we got from the ChromaDB are actually relevant to the User Input or not.
    try:
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

        logger.info(f"Selected {len(filtered_1)} relevant queries out of {len(relevant_queries)} after filtration part-1.")
    except Exception as e:
        log_and_notify_exception(e, "Unable to process the Part-1 of the Filteration Layer. Please check the logs for more details.")
        filtered_1 = relevant_queries

    # Part-2: De-duplication check for the filtered relevant queries. Such that, no repetitive queries are used for context building.
    filtered_2 = deduplicate_queries(filtered_1, relevant_query_tokens)
    if len(filtered_1) != len(filtered_2):
        logger.info(f"Selected {len(filtered_2)} relevant queries out of {len(filtered_1)} after filtration part-2.")

    return filtered_2


def summarize_context(context_list):

    text = " ".join(context_list)

    try:
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    except Exception as e:
        log_and_notify_exception(e, "Unable to Summarize Context at the moment. Please check the logs for more details.")
        summary = text

    return summary


def build_prompt_with_summary(user_input, n_results=10):

    try:
        print("Accessing Historical Chats...")
        results = collection.query(
            query_embeddings=embedder.encode([user_input], show_progress_bar=False).tolist(),
            n_results=n_results
        )
        
        raw_contexts = results['documents'][0] if results['documents'] else []
    except Exception as e:
        log_and_notify_exception(e, "Unable to Access Historical Chats at the moment. Please check logs for more details.")
        raw_contexts = []

    if len(raw_contexts) > 0:
        logger.info("Filtering Relevant Queries...")
        raw_contexts = filter_relevant_queries(user_input=user_input, relevant_queries=raw_contexts)

    logger.info("Summarizing Context...")
    summarized_context = summarize_context(raw_contexts) if raw_contexts else ""

    return summarized_context
