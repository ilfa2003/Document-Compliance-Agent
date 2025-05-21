#vector-index.py
import os
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel
import time

# Load environment variables (requires python-dotenv package)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Continue without dotenv if not installed

# Get MongoDB URI from environment variables
mongo_uri = os.getenv("MONGO_URI")

# Connect to MongoDB with error handling
try:
    client = MongoClient(mongo_uri)
    client.admin.command('ping')  # Test connection
    print("Connected to MongoDB!")
except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)

# Configure vector index parameters
search_index_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",  # Your embedding field name
                "numDimensions": 3072,  # Match your embedding model
                "similarity": "cosine"  # Preferred similarity metric
            }
        ]
    },
    name="invoice_vector_index",
    type="vectorSearch"
)

# Create the index
database = client["document_compliance"] #change to your database name
collection = database["invoices-test"] #change to your collection name

try:
    result = collection.create_search_index(model=search_index_model)
    print(f"Index {result} creation started...")
    
    # Wait for index completion
    print("Polling index status (max 60s):")
    start_time = time.time()
    while time.time() - start_time < 60:
        index_status = collection.list_search_indexes(name=result).next()
        if index_status.get('queryable', False):
            print("Index is ready!")
            break
        time.sleep(5)
    else:
        print("Timeout waiting for index creation")
        
except Exception as e:
    print(f"Index creation failed: {e}")
finally:
    client.close()
