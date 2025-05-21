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

def extract_western_express_data(pdf_path: str) -> Dict[str, Any]:
    """Extract data from a Western Express waybill/bill of lading PDF using Gemini Vision API"""
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
            result = process_western_express_with_gemini(img_data)
            all_results.append(result)
        doc.close()
    combined_data = combine_page_results(all_results)
    combined_data["extraction_timestamp"] = datetime.now().isoformat()
    combined_data["source_filename"] = os.path.basename(pdf_path)
    combined_data["document_type"] = "waybill"
    combined_data["carrier"] = "Western Express"
    return combined_data

def process_western_express_with_gemini(image_base64: str) -> Dict[str, Any]:
    """Process a Western Express waybill image with Gemini Vision API to extract data"""
    system_message = SystemMessage(content="""
    You are a specialized logistics document data extractor. Extract all relevant information from this Western Express waybill including:
    - Consignment number (CRN No.)
    - Shipper details (name, address, city, country, contact name, telephone)
    - Consignee details (name, address, city, country, contact name, telephone)
    - Number of packages
    - Total volume
    - Weight
    - Description of goods
    - Special instructions
    - Vehicle type
    - Payment details (shipper account, consignee account, etc.)
    - Collected by (date, time, signature)
    - Delivered by (date, time, signature)
    - Tracking information
    
    Format your response as a clean, properly formatted JSON object. Be precise and accurate.
    Return ONLY the JSON object, nothing else.
    """)
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all information from this Western Express waybill."},
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
        print(f"Error processing Western Express waybill with Gemini: {e}")
        print(f"Response content: {response.content[:500]}...")
        return {"error": str(e), "partial_response": response.content[:500]}

def combine_page_results(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine results from multiple pages into a single coherent result"""
    if not page_results:
        return {}
    combined = page_results[0]
    for page_data in page_results[1:]:
        # For waybills there are typically no line items, but we'll leave this logic
        # in case there are multiple package entries
        if "packages" in page_data and isinstance(page_data["packages"], list):
            if "packages" not in combined:
                combined["packages"] = []
            combined["packages"].extend(page_data["packages"])
    return combined

def get_waybill_text_for_embedding(waybill_data: dict) -> str:
    """Return a string with all waybill info except extraction_timestamp and source_filename"""
    filtered = {k: v for k, v in waybill_data.items() if k not in ["extraction_timestamp", "source_filename"]}
    return json.dumps(filtered, ensure_ascii=False, indent=2)

#You can change the collection name to any other name.
def save_to_mongodb_with_embedding(data: dict, embedding: list, collection_name: str = "invoices-test") -> str:
    """Save the extracted Western Express waybill data and embedding to MongoDB"""
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
    """Save the extracted Western Express waybill data to MongoDB"""
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

def process_and_save_western_express(pdf_path: str, add_embedding: bool = False) -> Dict[str, Any]:
    """Process a Western Express waybill PDF and save to MongoDB, with optional Gemini embedding"""
    try:
        print(f"Extracting data from Western Express waybill: {pdf_path}...")
        waybill_data = extract_western_express_data(pdf_path)
        if "error" in waybill_data:
            print(f"Error in extraction: {waybill_data['error']}")
            return {"status": "error", "message": f"Extraction failed: {waybill_data['error']}"}
        
        embedding = None
        if add_embedding:
            waybill_text = get_waybill_text_for_embedding(waybill_data)
            embedding = embeddings_model.embed_query(waybill_text)
            print("Saving to MongoDB with embedding...")
            doc_id = save_to_mongodb_with_embedding(waybill_data, embedding)
        else:
            print("Saving to MongoDB...")
            doc_id = save_to_mongodb(waybill_data)
            
        if doc_id:
            return {
                "status": "success",
                "message": "Western Express waybill processed and saved successfully",
                "mongodb_id": doc_id,
                "consignment_number": waybill_data.get("consignment_number", "Unknown"),
                "data": waybill_data
            }
        else:
            return {"status": "error", "message": "Failed to save to MongoDB"}
    except Exception as e:
        print(f"Error processing Western Express waybill: {e}")
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
    western_express_path = "truck consignment.pdf"  # Replace with your actual file path
    ADD_EMBEDDING = True  # Set to True to add Gemini embedding
    
    if not check_mongodb_connection():
        print("Warning: MongoDB is not available. Will extract data but not save.")
        waybill_data = extract_western_express_data(western_express_path)
        print("Extracted data:")
        print(json.dumps(waybill_data, indent=2))
    else:
        result = process_and_save_western_express(western_express_path, add_embedding=ADD_EMBEDDING)
        if result["status"] == "success":
            print(f"Successfully processed Western Express waybill {result['consignment_number']}")
            print(f"MongoDB ID: {result['mongodb_id']}")
        else:
            print(f"Error: {result['message']}")