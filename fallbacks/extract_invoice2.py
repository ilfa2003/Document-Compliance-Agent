import os
import base64
import json
import tempfile
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any, List

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import fitz  

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGODB_URI = os.getenv("MONGO_URI")

# Initialize Gemini model for vision
vision_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# Initialize Gemini embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",  # Or latest model as needed
    task_type="RETRIEVAL_DOCUMENT"
)

def extract_leminar_invoice_data(pdf_path: str) -> Dict[str, Any]:
    """Extract data from a Leminar Air Conditioning invoice PDF using Gemini Vision API"""
    with tempfile.TemporaryDirectory() as temp_dir:
        doc = fitz.open(pdf_path)
        all_results = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_path = os.path.join(temp_dir, f"page_{page_num}.png")
            pix.save(img_path)
            with open(img_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")
            result = process_leminar_invoice_with_gemini(img_data)
            all_results.append(result)
        doc.close()
    combined_data = combine_page_results(all_results)
    combined_data["extraction_timestamp"] = datetime.now().isoformat()
    combined_data["source_filename"] = os.path.basename(pdf_path)
    return combined_data

def process_leminar_invoice_with_gemini(image_base64: str) -> Dict[str, Any]:
    """Process a Leminar invoice image with Gemini Vision API to extract data"""
    system_message = SystemMessage(content="""
    You are a specialized HVAC invoice data extractor. Extract all relevant information from this Leminar Air Conditioning Company invoice including:
    - Invoice number (format: LACO-XX)
    - Invoice date
    - Date of export
    - Currency (AED)
    - Shipper/Exporter details (company name, address, contact)
    - Consignee details
    - Line items (each with SR No, description, HS code, origin, weight in KG, quantity, unit value, total value)
    - Sub-total
    - Total weight
    - Total amount in numbers and words
    - LAC reference numbers
    - Total packages
    
    Format your response as a clean, properly formatted JSON object. Be precise and accurate.
    Return ONLY the JSON object, nothing else.
    """)
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all information from this Leminar Air Conditioning invoice."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]
    )
    try:
        response = vision_model.invoke([system_message, human_message])
        response_text = response.content
        import re
        # Clean up the response to ensure it's valid JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        if not response_text.strip().startswith('{'):
            first_brace = response_text.find('{')
            if first_brace != -1:
                response_text = response_text[first_brace:]
        if not response_text.strip().endswith('}'):
            last_brace = response_text.rfind('}')
            if last_brace != -1:
                response_text = response_text[:last_brace+1]
        extracted_data = json.loads(response_text)
        return extracted_data
    except Exception as e:
        print(f"Error processing Leminar invoice with Gemini: {e}")
        print(f"Response content: {response.content[:500]}...")
        return {"error": str(e), "partial_response": response.content[:500]}

def combine_page_results(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine results from multiple pages into a single coherent result"""
    if not page_results:
        return {}
    combined = page_results[0]
    for page_data in page_results[1:]:
        if "line_items" in page_data and isinstance(page_data["line_items"], list):
            if "line_items" not in combined:
                combined["line_items"] = []
            combined["line_items"].extend(page_data["line_items"])
    return combined

def get_invoice_text_for_embedding(invoice_data: dict) -> str:
    """Return a string with all invoice info except extraction_timestamp and source_filename"""
    filtered = {k: v for k, v in invoice_data.items() if k not in ["extraction_timestamp", "source_filename"]}
    return json.dumps(filtered, ensure_ascii=False, indent=2)

#You can change the collection name to any other name.
def save_to_mongodb_with_embedding(data: dict, embedding: list, collection_name: str = "invoices-test") -> str:
    """Save the extracted Leminar invoice data and embedding to MongoDB"""
    try:
        client = MongoClient(MONGODB_URI)
        db = client.document_compliance #You can change the database name to any other name.
        collection = db[collection_name]
        data["embedding"] = embedding
        result = collection.insert_one(data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return None
    finally:
        if 'client' in locals():
            client.close()

#You can change the collection name to any other name.
def save_to_mongodb(data: Dict[str, Any], collection_name: str = "invoices-test") -> str:
    """Save the extracted Leminar invoice data to MongoDB"""
    try:
        client = MongoClient(MONGODB_URI)
        db = client.document_compliance #You can change the database name to any other name.
        collection = db[collection_name]
        result = collection.insert_one(data)
        return str(result.inserted_id)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")
        return None
    finally:
        if 'client' in locals():
            client.close()

def process_and_save_leminar_invoice(pdf_path: str, add_embedding: bool = False) -> Dict[str, Any]:
    """Process a Leminar invoice PDF and save to MongoDB, with optional Gemini embedding"""
    try:
        print(f"Extracting data from Leminar invoice: {pdf_path}...")
        invoice_data = extract_leminar_invoice_data(pdf_path)
        if "error" in invoice_data:
            print(f"Error in extraction: {invoice_data['error']}")
            return {"status": "error", "message": f"Extraction failed: {invoice_data['error']}"}
        
        # Additional processing specific to Leminar invoices
        if "line_items" in invoice_data:
            total_amount = sum(item.get("total_value", 0) for item in invoice_data["line_items"])
            invoice_data["calculated_total"] = total_amount
            
        embedding = None
        if add_embedding:
            invoice_text = get_invoice_text_for_embedding(invoice_data)
            embedding = embeddings_model.embed_query(invoice_text)
            print("Saving to MongoDB with embedding...")
            doc_id = save_to_mongodb_with_embedding(invoice_data, embedding)
        else:
            print("Saving to MongoDB...")
            doc_id = save_to_mongodb(invoice_data)
            
        if doc_id:
            return {
                "status": "success",
                "message": "Leminar invoice processed and saved successfully",
                "mongodb_id": doc_id,
                "invoice_number": invoice_data.get("invoice_number", "Unknown"),
                "data": invoice_data
            }
        else:
            return {"status": "error", "message": "Failed to save to MongoDB"}
    except Exception as e:
        print(f"Error processing Leminar invoice: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"Processing failed: {str(e)}"}

def check_mongodb_connection() -> bool:
    """Check if MongoDB is available"""
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.server_info()
        client.close()
        return True
    except Exception as e:
        print(f"MongoDB connection error: {e}")
        return False

def find_similar_invoices(invoice_data: dict, similarity_threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Find similar invoices in the database using embedding similarity"""
    if not check_mongodb_connection():
        return []
        
    try:
        # Extract key text for embedding
        invoice_text = get_invoice_text_for_embedding(invoice_data)
        embedding = embeddings_model.embed_query(invoice_text)
        
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client.document_compliance
        collection = db.invoices
        
        # Use the $vectorSearch aggregation pipeline 
        # This assumes MongoDB has vector search capability
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "invoice_embedding_index",
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 100,
                    "limit": 5
                }
            },
            {
                "$project": {
                    "invoice_number": 1,
                    "invoice_date": 1, 
                    "total_amount": 1,
                    "shipper": 1,
                    "consignee": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        similar_docs = list(collection.aggregate(pipeline))
        
        # Filter by threshold
        return [doc for doc in similar_docs if doc.get("score", 0) > similarity_threshold]
    except Exception as e:
        print(f"Error finding similar invoices: {e}")
        return []
    finally:
        if 'client' in locals():
            client.close()

def analyze_invoice_trends(date_range: Dict[str, str] = None) -> Dict[str, Any]:
    """Analyze Leminar invoice data to identify trends and patterns"""
    if not check_mongodb_connection():
        return {"status": "error", "message": "MongoDB not available"}
        
    try:
        client = MongoClient(MONGODB_URI)
        db = client.document_compliance
        collection = db.invoices
        
        # Build query based on date range if provided
        query = {}
        if date_range:
            query["invoice_date"] = {
                "$gte": date_range.get("start"),
                "$lte": date_range.get("end")
            }
        
        # Aggregation for total invoice amount by month
        monthly_totals_pipeline = [
            {"$match": query},
            {"$group": {
                "_id": {
                    "year": {"$year": {"$dateFromString": {"dateString": "$invoice_date"}}},
                    "month": {"$month": {"$dateFromString": {"dateString": "$invoice_date"}}}
                },
                "total_amount": {"$sum": "$total_amount"},
                "count": {"$sum": 1}
            }},
            {"$sort": {"_id.year": 1, "_id.month": 1}}
        ]
        
        # Aggregation for top products by quantity
        top_products_pipeline = [
            {"$match": query},
            {"$unwind": "$line_items"},
            {"$group": {
                "_id": "$line_items.description",
                "total_quantity": {"$sum": "$line_items.quantity"},
                "total_value": {"$sum": "$line_items.total_value"}
            }},
            {"$sort": {"total_quantity": -1}},
            {"$limit": 10}
        ]
        
        # Aggregation for top countries of origin
        origin_pipeline = [
            {"$match": query},
            {"$unwind": "$line_items"},
            {"$group": {
                "_id": "$line_items.origin",
                "total_quantity": {"$sum": "$line_items.quantity"},
                "item_count": {"$sum": 1}
            }},
            {"$sort": {"total_quantity": -1}}
        ]
        
        monthly_totals = list(collection.aggregate(monthly_totals_pipeline))
        top_products = list(collection.aggregate(top_products_pipeline))
        origins = list(collection.aggregate(origin_pipeline))
        
        return {
            "status": "success",
            "monthly_totals": monthly_totals,
            "top_products": top_products,
            "origins": origins,
            "total_invoices": collection.count_documents(query)
        }
        
    except Exception as e:
        print(f"Error analyzing invoice trends: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    leminar_invoice_path = "invoice.pdf"  # Replace with your actual file path
    ADD_EMBEDDING = True  # Set to True to add Gemini embedding
    
    if not check_mongodb_connection():
        print("Warning: MongoDB is not available. Will extract data but not save.")
        invoice_data = extract_leminar_invoice_data(leminar_invoice_path)
        print("Extracted data:")
        print(json.dumps(invoice_data, indent=2))
    else:
        result = process_and_save_leminar_invoice(leminar_invoice_path, add_embedding=ADD_EMBEDDING)
        if result["status"] == "success":
            print(f"Successfully processed Leminar invoice {result['invoice_number']}")
            print(f"MongoDB ID: {result['mongodb_id']}")
            
            # Find similar invoices
            similar = find_similar_invoices(result["data"])
            if similar:
                print("\nSimilar invoices found:")
                for doc in similar:
                    print(f"- Invoice {doc['invoice_number']} dated {doc['invoice_date']} (Similarity: {doc['score']:.2f})")
            
            # Run trend analysis for the last 6 months
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            
            print("\nRunning trend analysis for the last 6 months...")
            trends = analyze_invoice_trends({"start": start_date, "end": end_date})
            if trends["status"] == "success":
                print(f"Total invoices in period: {trends['total_invoices']}")
                print("Top products:")
                for product in trends["top_products"][:3]:
                    print(f"- {product['_id']}: {product['total_quantity']} units, {product['total_value']} AED")
        else:
            print(f"Error: {result['message']}")