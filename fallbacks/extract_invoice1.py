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
import fitz  # PyMuPDF

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

def extract_invoice_data_from_pdf(pdf_path: str) -> Dict[str, Any]:
    """Extract invoice data from a PDF using Gemini Vision API"""
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
            result = process_image_with_gemini(img_data)
            all_results.append(result)
        doc.close()
    combined_data = combine_page_results(all_results)
    combined_data["extraction_timestamp"] = datetime.now().isoformat()
    combined_data["source_filename"] = os.path.basename(pdf_path)
    return combined_data

def process_image_with_gemini(image_base64: str) -> Dict[str, Any]:
    """Process an image with Gemini Vision API to extract invoice data"""
    system_message = SystemMessage(content="""
    You are a specialized invoice data extractor. Extract all relevant information from this invoice including:
    - Invoice number
    - Invoice date
    - Customer number
    - Customer name
    - Bill to address
    - PO number
    - Due date
    - Currency
    - Line items (each with description, quantity, unit price, and total)
    - Subtotal
    - Tax amount
    - Total amount
    - Payment terms
    Format your response as a clean, properly formatted JSON object. Be precise and accurate.
    Return ONLY the JSON object, nothing else.
    """)
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all information from this invoice document."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
        ]
    )
    try:
        response = vision_model.invoke([system_message, human_message])
        response_text = response.content
        import re
        json_match = re.search(r'``````', response_text, re.DOTALL)
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
        print(f"Error processing image with Gemini: {e}")
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
    """Save the extracted invoice data and embedding to MongoDB"""
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
    """Save the extracted invoice data to MongoDB"""
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

def process_and_save_invoice(pdf_path: str, add_embedding: bool = False) -> Dict[str, Any]:
    """Process an invoice PDF and save to MongoDB, with optional Gemini embedding"""
    try:
        print(f"Extracting data from {pdf_path}...")
        invoice_data = extract_invoice_data_from_pdf(pdf_path)
        if "error" in invoice_data:
            print(f"Error in extraction: {invoice_data['error']}")
            return {"status": "error", "message": f"Extraction failed: {invoice_data['error']}"}
        embedding = None
        if add_embedding:
            invoice_text = get_invoice_text_for_embedding(invoice_data)
            embedding = embeddings_model.embed_query(invoice_text)
        if add_embedding:
            print("Saving to MongoDB with embedding...")
            doc_id = save_to_mongodb_with_embedding(invoice_data, embedding)
        else:
            print("Saving to MongoDB...")
            doc_id = save_to_mongodb(invoice_data)
        if doc_id:
            return {
                "status": "success",
                "message": "Invoice processed and saved successfully",
                "mongodb_id": doc_id,
                "invoice_number": invoice_data.get("invoice_number", "Unknown"),
                "data": invoice_data
            }
        else:
            return {"status": "error", "message": "Failed to save to MongoDB"}
    except Exception as e:
        print(f"Error processing invoice: {e}")
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
    invoice_pdf_path = "invoice (1).PDF"  # Replace with your actual file path
    ADD_EMBEDDING = True  # Set to True to add Gemini embedding
    if not check_mongodb_connection():
        print("Warning: MongoDB is not available. Will extract data but not save.")
        invoice_data = extract_invoice_data_from_pdf(invoice_pdf_path)
        print("Extracted data:")
        print(json.dumps(invoice_data, indent=2))
    else:
        result = process_and_save_invoice(invoice_pdf_path, add_embedding=ADD_EMBEDDING)
        if result["status"] == "success":
            print(f"Successfully processed invoice {result['invoice_number']}")
            print(f"MongoDB ID: {result['mongodb_id']}")
        else:
            print(f"Error: {result['message']}")
