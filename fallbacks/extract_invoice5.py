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

def extract_customs_declaration_data(pdf_path: str) -> Dict[str, Any]:
    """Extract data from a UAE Federal Customs Authority declaration PDF using Gemini Vision API"""
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
            result = process_customs_declaration_with_gemini(img_data)
            all_results.append(result)
        doc.close()
    combined_data = combine_page_results(all_results)
    combined_data["extraction_timestamp"] = datetime.now().isoformat()
    combined_data["source_filename"] = os.path.basename(pdf_path)
    combined_data["document_type"] = "customs_declaration"
    combined_data["issuing_authority"] = "UAE Federal Customs Authority"
    return combined_data

def process_customs_declaration_with_gemini(image_base64: str) -> Dict[str, Any]:
    """Process a UAE Customs declaration image with Gemini Vision API to extract data"""
    system_message = SystemMessage(content="""
    You are a specialized customs document data extractor. Extract all relevant information from this UAE Federal Customs Authority declaration including:
    - Declaration number (DEC NO.)
    - Declaration date
    - Port type
    - Declaration type (usually EXPORT)
    - Customs declaration statistical information
    - Net weight and gross weight
    - Consignee/Exporter details
    - Commercial registration numbers
    - Number of packages
    - Marks and numbers
    - Port of loading
    - Port of discharge
    - Destination
    - Line items (each with HS code, goods description, origin, CIF values)
    - Total duty
    - Clearing agent details
    - Payment information
    - Exit port
    - Release date
    - Any reference numbers to invoices or other documents
    
    Format your response as a clean, properly formatted JSON object. Be precise and accurate.
    Return ONLY the JSON object, nothing else.
    """)
    human_message = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all information from this UAE Federal Customs Authority declaration."},
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
        print(f"Error processing Customs Declaration with Gemini: {e}")
        print(f"Response content: {response.content[:500]}...")
        return {"error": str(e), "partial_response": response.content[:500]}

def combine_page_results(page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine results from multiple pages into a single coherent result"""
    if not page_results:
        return {}
    combined = page_results[0]
    for page_data in page_results[1:]:
        # For customs declarations, there might be multiple line items
        if "line_items" in page_data and isinstance(page_data["line_items"], list):
            if "line_items" not in combined:
                combined["line_items"] = []
            combined["line_items"].extend(page_data["line_items"])
    return combined

def get_declaration_text_for_embedding(declaration_data: dict) -> str:
    """Return a string with all declaration info except extraction_timestamp and source_filename"""
    filtered = {k: v for k, v in declaration_data.items() if k not in ["extraction_timestamp", "source_filename"]}
    return json.dumps(filtered, ensure_ascii=False, indent=2)

#You can change the collection name to any other name.
def save_to_mongodb_with_embedding(data: dict, embedding: list, collection_name: str = "invoices-test") -> str:
    """Save the extracted UAE Customs declaration data and embedding to MongoDB"""
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
    """Save the extracted UAE Customs declaration data to MongoDB"""
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

def process_and_save_customs_declaration(pdf_path: str, add_embedding: bool = False) -> Dict[str, Any]:
    """Process a UAE Customs declaration PDF and save to MongoDB, with optional Gemini embedding"""
    try:
        print(f"Extracting data from UAE Customs declaration: {pdf_path}...")
        declaration_data = extract_customs_declaration_data(pdf_path)
        if "error" in declaration_data:
            print(f"Error in extraction: {declaration_data['error']}")
            return {"status": "error", "message": f"Extraction failed: {declaration_data['error']}"}
        
        # Calculate total values if line items are present
        if "line_items" in declaration_data and isinstance(declaration_data["line_items"], list):
            total_value = sum(item.get("cif_local_value", 0) for item in declaration_data["line_items"])
            declaration_data["calculated_total_value"] = total_value
        
        # Add relationships to associated documents if present
        related_docs = []
        # Check for invoice references
        if "invoice_reference" in declaration_data:
            related_docs.append({"type": "invoice", "document_id": declaration_data["invoice_reference"]})
        # Check for bill of lading/airway bill references
        if "awb_number" in declaration_data:
            related_docs.append({"type": "airway_bill", "document_id": declaration_data["awb_number"]})
        
        if related_docs:
            declaration_data["related_documents"] = related_docs
            
        embedding = None
        if add_embedding:
            declaration_text = get_declaration_text_for_embedding(declaration_data)
            embedding = embeddings_model.embed_query(declaration_text)
            print("Saving to MongoDB with embedding...")
            doc_id = save_to_mongodb_with_embedding(declaration_data, embedding)
        else:
            print("Saving to MongoDB...")
            doc_id = save_to_mongodb(declaration_data)
            
        if doc_id:
            return {
                "status": "success",
                "message": "UAE Customs declaration processed and saved successfully",
                "mongodb_id": doc_id,
                "declaration_number": declaration_data.get("declaration_number", "Unknown"),
                "data": declaration_data
            }
        else:
            return {"status": "error", "message": "Failed to save to MongoDB"}
    except Exception as e:
        print(f"Error processing UAE Customs declaration: {e}")
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

def extract_related_documents(declaration_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract references to related documents from the declaration data"""
    related_docs = []
    
    # Look for invoice numbers in Other Remarks section
    if "other_remarks" in declaration_data:
        remarks = declaration_data["other_remarks"]
        import re
        # Look for patterns like LACO-39 or similar invoice references
        invoice_match = re.search(r'[A-Z]+-\d+', remarks)
        if invoice_match:
            related_docs.append({
                "type": "invoice", 
                "document_id": invoice_match.group(0)
            })
    
    # Check marks and numbers for container references
    if "marks_and_numbers" in declaration_data:
        marks = declaration_data["marks_and_numbers"]
        if marks and "C.NOTE" in marks:
            # Extract C.NOTE reference
            import re
            cnote_match = re.search(r'C\.NOTE\s+(\d+)', marks)
            if cnote_match:
                related_docs.append({
                    "type": "consignment_note", 
                    "document_id": cnote_match.group(1)
                })
    
    return related_docs

if __name__ == "__main__":
    customs_declaration_path = "customs.pdf"  # Replace with actual file path
    ADD_EMBEDDING = True  # Set to True to add Gemini embedding
    
    if not check_mongodb_connection():
        print("Warning: MongoDB is not available. Will extract data but not save.")
        declaration_data = extract_customs_declaration_data(customs_declaration_path)
        print("Extracted data:")
        print(json.dumps(declaration_data, indent=2))
    else:
        result = process_and_save_customs_declaration(customs_declaration_path, add_embedding=ADD_EMBEDDING)
        if result["status"] == "success":
            print(f"Successfully processed UAE Customs declaration {result['declaration_number']}")
            print(f"MongoDB ID: {result['mongodb_id']}")
            
            # Check if there are related documents
            if "related_documents" in result["data"]:
                print("\nRelated documents:")
                for doc in result["data"]["related_documents"]:
                    print(f"- {doc['type'].capitalize()} {doc['document_id']}")
            
            # Display line items summary
            if "line_items" in result["data"]:
                print("\nLine items summary:")
                for i, item in enumerate(result["data"]["line_items"], 1):
                    print(f"{i}. {item.get('goods_description', 'Unknown')} - {item.get('origin', 'Unknown')} - AED {item.get('cif_local_value', 0)}")
        else:
            print(f"Error: {result['message']}")