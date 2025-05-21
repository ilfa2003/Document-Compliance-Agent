import os
import re
import json
from pdf2image import convert_from_path
import pytesseract
from PIL import Image

# Configure paths
POPPLER_PATH = r"C:\Users\ilfas\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update with your Tesseract path

# Configure executables
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

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

def process_pdf(pdf_path):
    # Convert PDF to images
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)
    
    results = []
    
    for idx, image in enumerate(images):
        print(f"\nProcessing page {idx+1}")
        
        # Preprocess image
        gray = image.convert('L')  # Convert to grayscale
        sharp = gray.point(lambda x: 0 if x < 200 else 255)  # Simple thresholding
        
        # Perform OCR
        ocr_text = pytesseract.image_to_string(sharp, config='--psm 6')
        print("\nRaw OCR Output:")
        print("-"*40)
        print(ocr_text)
        
        # Extract structured data
        structured_data = extract_structured_data(ocr_text)
        results.append(structured_data)
        
        print("\nStructured Data:")
        print(json.dumps(structured_data, indent=2))
    
    return results

if __name__ == "__main__":
    PDF_PATH = "truck consignment, exit certificate, delivery note-pages-1.pdf"
    process_pdf(PDF_PATH)
