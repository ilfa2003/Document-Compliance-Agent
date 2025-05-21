# Updated vector-search.py
import os
import pymongo
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables with validation
load_dotenv()

# Validate required environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")
if not MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables")

# Initialize Gemini embeddings with explicit API key
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    task_type="RETRIEVAL_QUERY",
    google_api_key=GOOGLE_API_KEY  # Explicit API key setting
)

# Generate embedding for search term
search_term = "travel"
query_vector = embeddings.embed_query(search_term)

# MongoDB connection with improved error handling
DATABASE_NAME = "document_compliance"
COLLECTION_NAME = "invoices"

try:
    client = pymongo.MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000  # 5-second timeout
    )
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # Verify connection and collection existence
    client.admin.command('ping')
    if COLLECTION_NAME not in db.list_collection_names():
        raise ValueError(f"Collection {COLLECTION_NAME} not found")
    
    print("Successfully connected to MongoDB!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)

# Vector search pipeline with all fields
pipeline = [
    {
        "$vectorSearch": {
            "index": "invoice_vector_index",
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": 200,
            "limit": 10
        }
    },
    {
        "$addFields": {
            "score": {"$meta": "vectorSearchScore"}
        }
    },
    {
        "$project": {
            "embedding": 0,  # Exclude the embedding field
            # Optionally, include only the fields you want, or omit to keep all except embedding
        }
    },
    {
        "$match": {
            "score": {"$gte": 0.76}
        }
    }
]


try:
    results = collection.aggregate(pipeline)
    print(f"\nSearch results for '{search_term}':")
    for doc in results:
        print("-" * 50)
        print(f"Score: {doc.get('score', 0):.4f}")
        print("Document:")
        print(json.dumps(doc, indent=2, default=str))
except Exception as e:
    print(f"Search failed: {e}")
finally:
    client.close()
