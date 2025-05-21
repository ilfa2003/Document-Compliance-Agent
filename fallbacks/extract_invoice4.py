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

def extract_customs_certificate_data(pdf_path: str) -> Dict[str, Any]:
    """Extract data from a Dubai Customs Exit/Entry Certificate PDF using Gemini Vision API"""
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
            result = process_customs_certificate_with_gemini(img_data)
            all_results.append(result)
        doc.close()
    combined_data = combine_page_results(all_results)
    combined_data["extraction_timestamp"] = datetime.now().isoformat()
    combined_data["source_filename"] = os.path.basename(pdf_path)
    combined_data["document_type"] = "customs_certificate"
    combined_data["issuing_authority"] = "Dubai Customs"
    return combined_data

def process_customs_certificate_with_gemini(image_base64: str) -> Dict[str, Any]:
    """Process a Dubai Customs certificate image with Gemini Vision API to extract data"""
    system_message = SystemMessage(content="""
    You are a specialized customs document data extractor. Extract all relevant information from this Dubai Customs Exit/Entry Certificate including:
    - Certificate date
    - Exporter name and details
    - Bill number
    - Export bill/Air way bill/Manifest information
    - Country of origin
    - Point of exit
    - Destination
    - Description of goods (all items listed)
    - Invoice details (invoice number, date)
    - Total quantity
    - Total weight
    - Container/Vehicle number
    - Customs seal number
    - Any reference to related invoices
    
    Format your response as a clean, properly formatted JSON object. Be precise and accurate.
    Return ONLY the JSON object, nothing else.
    """)
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all information from this Dubai Customs Exit/Entry Certificate."},
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
        print(f"Error processing Customs Certificate with Gemini: {e}")
        print(f"Response content: {response.content[:500]}...")
        return {"error": str(e), "partial_response": response.content[:500]}

def combine_page_results(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine results from multiple pages into a single coherent result"""
    if not page_results:
        return {}
    combined = page_results[0]
    for page_data in page_results[1:]:
        # For customs certificates, there might be multiple goods entries
        if "goods" in page_data and isinstance(page_data["goods"], list):
            if "goods" not in combined:
                combined["goods"] = []
            combined["goods"].extend(page_data["goods"])
    return combined

def get_certificate_text_for_embedding(certificate_data: dict) -> str:
    """Return a string with all certificate info except extraction_timestamp and source_filename"""
    filtered = {k: v for k, v in certificate_data.items() if k not in ["extraction_timestamp", "source_filename"]}
    return json.dumps(filtered, ensure_ascii=False, indent=2)

#You can change the collection name to any other name.
def save_to_mongodb_with_embedding(data: dict, embedding: list, collection_name: str = "invoices-test") -> str:
    """Save the extracted Dubai Customs certificate data and embedding to MongoDB"""
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
    """Save the extracted Dubai Customs certificate data to MongoDB"""
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

def process_and_save_customs_certificate(pdf_path: str, add_embedding: bool = False) -> Dict[str, Any]:
    """Process a Dubai Customs certificate PDF and save to MongoDB, with optional Gemini embedding"""
    try:
        print(f"Extracting data from Dubai Customs certificate: {pdf_path}...")
        certificate_data = extract_customs_certificate_data(pdf_path)
        if "error" in certificate_data:
            print(f"Error in extraction: {certificate_data['error']}")
            return {"status": "error", "message": f"Extraction failed: {certificate_data['error']}"}
        
        # Add relationships to associated invoices if present
        if "invoice_number" in certificate_data:
            certificate_data["related_documents"] = [
                {"type": "invoice", "document_id": certificate_data["invoice_number"]}
            ]
            
        embedding = None
        if add_embedding:
            certificate_text = get_certificate_text_for_embedding(certificate_data)
            embedding = embeddings_model.embed_query(certificate_text)
            print("Saving to MongoDB with embedding...")
            doc_id = save_to_mongodb_with_embedding(certificate_data, embedding)
        else:
            print("Saving to MongoDB...")
            doc_id = save_to_mongodb(certificate_data)
            
        if doc_id:
            return {
                "status": "success",
                "message": "Dubai Customs certificate processed and saved successfully",
                "mongodb_id": doc_id,
                "bill_number": certificate_data.get("bill_number", "Unknown"),
                "data": certificate_data
            }
        else:
            return {"status": "error", "message": "Failed to save to MongoDB"}
    except Exception as e:
        print(f"Error processing Dubai Customs certificate: {e}")
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

if __name__ == "__main__":
    customs_certificate_path = "customs-certificate.pdf"  # Replace with actual file path
    ADD_EMBEDDING = True  # Set to True to add Gemini embedding
    
    if not check_mongodb_connection():
        print("Warning: MongoDB is not available. Will extract data but not save.")
        certificate_data = extract_customs_certificate_data(customs_certificate_path)
        print("Extracted data:")
        print(json.dumps(certificate_data, indent=2))
    else:
        result = process_and_save_customs_certificate(customs_certificate_path, add_embedding=ADD_EMBEDDING)
        if result["status"] == "success":
            print(f"Successfully processed Dubai Customs certificate {result['bill_number']}")
            print(f"MongoDB ID: {result['mongodb_id']}")
            
            # Check if there are related documents (like invoices) 
            if "related_documents" in result["data"]:
                print("\nRelated documents:")
                for doc in result["data"]["related_documents"]:
                    print(f"- {doc['type'].capitalize()} {doc['document_id']}")
        else:
            print(f"Error: {result['message']}")