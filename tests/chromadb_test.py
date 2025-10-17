from tqdm import tqdm
import json, chromadb, re, difflib
from nltk.corpus import stopwords
from transformers import pipeline
from sentence_transformers import SentenceTransformer

"""First Test - Creation | Sample Data | Data Embeddings | Data Insertion | Query on Similarity"""

# # Initializing the Embedding Model:
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # Initializing Chroma Client & Create/Get Collection:
# client = chromadb.Client()
# collection = client.get_or_create_collection(name="chat_history_test")
# print("ChromaDB Client & Collection Created.")

# # Sample Chat Snippets:
# chat_snippets = [
#     "Hello, how can I help you today?", 
#     "What is the weather like in Mumbai, India.", 
#     "Tell me a joke.", 
#     "How do I reset my password?"
# ]

# # Create Embeddings
# embeddings = embedder.encode(chat_snippets).tolist()

# # Insert embeddings and texts into the collection:
# collection.add(
#     documents=chat_snippets, 
#     embeddings=embeddings, 
#     metadatas=[{"source": "test"}]*len(chat_snippets), 
#     ids=[f"test_{i}" for i in range(len(chat_snippets))]
# )
# print("Data Insertion Completed.")

# # Query: find closest message to this input
# query = "How do I change my PASSWORD?"
# query_emb = embedder.encode([query]).tolist()

# results = collection.query(query_embeddings=query_emb, n_results=2)

# print("Search result for the query:", query)
# for i, doc in enumerate(results["documents"][0]):
#     print(f"{i+1}. {doc}")

# # Clean Up: delete the test records
# if input("Shall we proceed to delete the test records? ") == "yes":
#     collection.delete(ids=[f"test_{i}" for i in range(len(chat_snippets))])
#     print("Test Records Deleted")

# ------------------------------------------------------------------------------------------------------------------------

"""Data Check on ChromaDB:"""

# # From this Persistent Path, it will check if any collections are existing.
# client = chromadb.PersistentClient(path="ChromaDB_Storage")
# collection_name = "chat_history"
# print("Available collections:", client.list_collections())

# # Deleting a Collection:
# client.delete_collection(name=collection_name)
# print(f"Collection '{collection_name}' deleted successfully.")
# print("Available collections:", client.list_collections())

# # No. of entries in the Collection.
# collection = client.get_or_create_collection(name=collection_name)
# print(f"Collection '{collection_name}' count:", collection.count())

# # Retrieve all records in the collection
# entries = collection.get(include=["documents", "metadatas"])
# ids = entries["ids"]

# print("Data:\n")
# for idx, (doc, meta, id_) in enumerate(zip(entries["documents"], entries["metadatas"], ids)):
#     if idx > 10:
#         break
#     print(f"{idx+1}: ID={id_}, Index={meta['index']}, Chat: {doc}")

# ------------------------------------------------------------------------------------------------------------------------

"""Data Insertion in ChromaDB"""

# client = chromadb.PersistentClient(path="ChromaDB_Storage")
# collection_name = "chat_history"
# collection = client.get_or_create_collection(name=collection_name)
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# with open("Chat_History.json", "r") as f:
#     data = json.load(f)
# for idx, chat in tqdm(enumerate(data[collection_name])):
#     text = chat["human"]
#     embedding = embedder.encode([text]).tolist()
#     collection.add(
#         documents=[text],
#         embeddings=embedding,
#         metadatas=[{"import": "json", "index": (idx + 1)}],
#         ids=[f"chat_{(idx + 1)}"]
#     )

# print(f"Historical Chats inserted into ChromaDB's `{collection_name}` Collection.\n\
# List of available Collections: {client.list_collections()}\n\
# No. of entries in the Collection `{collection_name}`: {collection.count()}")

# ------------------------------------------------------------------------------------------------------------------------

"""Querying similar Chat Messages"""

# client = chromadb.PersistentClient(path="ChromaDB_Storage")
# collection = client.get_collection(name="chat_history")
# embedder = SentenceTransformer('all-MiniLM-L6-v2')

# # # Query: find closest message to this input
# # query = "Hey, tell me something about yourself."
# query = "How python, the programming language, works behind the IDEs. Keep it simple."
# query_emb = embedder.encode([query]).tolist()

# results = collection.query(query_embeddings=query_emb, n_results=10)

# print("Search result for the query:", query)
# for i, doc in enumerate(results["documents"][0]):
#     print(f"{i+1}. {doc}")

# summarizer = pipeline("summarization", model="t5-small")

# similar_message_history = " ".join(results["documents"][0])
# summary = summarizer(similar_message_history, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
# print("\n", summary)

# ------------------------------------------------------------------------------------------------------------------------

"""Creation & Testing of Final Filteration Layer"""


def clean_and_tokenize(text: str):
    clean_text = re.sub(r"\W+", " ", text.lower())
    tokens = [t for t in clean_text.split() if t not in stop_words]
    return set(tokens)


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


def deduplicate_queries(filtered_queries, tokenized_dict, threshold=0.7):
    """
    Deduplicate filtered queries by comparing their meaning via token overlap (Jaccard similarity).
    Prints the pairwise duplicate score for transparency.
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

    print(f"Selected {len(filtered_1)} relevant queries out of {len(relevant_queries)} after filtration part-1.\n")

    # Part-2: De-duplication check for the filtered relevant queries. Such that, no repetitive queries are used for context building.
    filtered_2 = deduplicate_queries(filtered_1, relevant_query_tokens)
    if len(filtered_1) != len(filtered_2):
        print(f"Selected {len(filtered_2)} relevant queries out of {len(filtered_1)} after filtration part-2.")

    return filtered_2

start_time = timer()
client = chromadb.PersistentClient(path="ChromaDB_Storage")
collection = client.get_collection(name="chat_history")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
stop_words = set(stopwords.words("english"))

# # Query: find closest message to this input
# query = "Hey, tell me something about yourself."
query = "How python, the programming language, works behind the IDEs. Keep it simple."
query_emb = embedder.encode([query]).tolist()

results = collection.query(query_embeddings=query_emb, n_results=10)


filtered_relevant_queries = filter_relevant_queries(query, results["documents"][0])


print("Filtered Relevant Queries:", query)
for i, doc in enumerate(filtered_relevant_queries):
    print(f"{i+1}. {doc}")

summarizer = pipeline("summarization", model="t5-small")

similar_message_history = " ".join(filtered_relevant_queries)
summary = summarizer(similar_message_history, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
print("\n", summary)
timer(start_time)
