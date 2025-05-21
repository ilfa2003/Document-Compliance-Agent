import os
import base64
import json
import re
from io import BytesIO
from pdf2image import convert_from_path
from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()

# Configure system paths (update with your Poppler path)
os.environ["PATH"] += os.pathsep + r"C:\Users\ilfas\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

# Set your Roboflow API key
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Path to your PDF file
PDF_PATH = "truck consignment, exit certificate, delivery note-pages-1.pdf"

# Convert PDF pages to images
images = convert_from_path(PDF_PATH, dpi=300)

def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def extract_structured_data(ocr_text):
    data = {}
    
    # Consignment Note Number
    cn_match = re.search(r"C/N No\.\s*:\s*(\S+)", ocr_text)
    if cn_match:
        data["Consignment Note Number"] = cn_match.group(1)
    
    # Shipper Information
    shipper_match = re.search(r"SHIPPER:\s*(.*?)\s*City:\s*(.*?)\s*Country:\s*(.*?)\s*Contact Name:\s*(.*?)\s*Telephone:\s*(\d+)", 
                            ocr_text, re.DOTALL)
    if shipper_match:
        data["Shipper"] = {
            "Name": shipper_match.group(1).strip(),
            "City": shipper_match.group(2).strip(),
            "Country": shipper_match.group(3).strip(),
            "Contact": shipper_match.group(4).strip(),
            "Phone": shipper_match.group(5).strip()
        }
    
    # Consignee Information
    consignee_match = re.search(r"CONSIGNEE \(Receiver\)\s*(.*?)\s*City:\s*(.*?)\s*Country:\s*(.*?)\s*Contact Name:\s*(.*?)\s*Telephone:\s*(\d+)", 
                              ocr_text, re.DOTALL)
    if consignee_match:
        data["Consignee"] = {
            "Name": consignee_match.group(1).strip(),
            "City": consignee_match.group(2).strip(),
            "Country": consignee_match.group(3).strip(),
            "Contact": consignee_match.group(4).strip(),
            "Phone": consignee_match.group(5).strip()
        }
    
    # Package Details
    data["Packages"] = {
        "No. of Packages": re_search(r"No\. of Packages:\s*(\S+)", ocr_text),
        "Gross Weight": re_search(r"Gross Wt\.\s*(\d+)", ocr_text),
        "Chargeable Weight": re_search(r"Chargeable Wt\.\s*(\d+)", ocr_text)
    }
    
    # Tracking Information
    data["Tracking"] = {
        "Booking No.": re_search(r"Booking No\.:\s*(\S+)", ocr_text),
        "Vehicle No.": re_search(r"Truck No\.:\s*(\S+)", ocr_text)
    }
    
    # Freight Charges
    freight_match = re.search(r"Freight Charge:\s*([\d.,]+)", ocr_text)
    if freight_match:
        data["Freight Charge"] = freight_match.group(1)
    
    return data

def re_search(pattern, text):
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None

# Process each image
# Process each image
for idx, image in enumerate(images):
    print(f"\n{'='*40}\nProcessing page {idx + 1}\n{'='*40}")
    
    # Convert image to base64
    image_base64 = pil_to_base64(image)
    
    payload = {
        "image": {"type": "base64", "value": image_base64}
    }
    
    response = requests.post(
        f"https://infer.roboflow.com/doctr/ocr?api_key={API_KEY}",
        json=payload
    )

    if response.status_code == 200:
        ocr_result = response.json()
        raw_text = ocr_result.get("result", "")
        
        # Display raw OCR output
        print("\nRaw OCR Text:")
        print("-"*40)
        print(raw_text)
        
        # Extract and display structured data
        structured_data = extract_structured_data(raw_text)
        print("\nStructured Data:")
        print("-"*40)
        print(json.dumps(structured_data, indent=2))
        
    else:
        print(f"Error processing page {idx+1}: {response.status_code}")
