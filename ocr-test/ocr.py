import easyocr
import os
import re
import json
from pdf2image import convert_from_path
import numpy as np

# Configure system paths (update with your Poppler path)
os.environ["PATH"] += os.pathsep + r"C:\Users\ilfas\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin"

def extract_structured_data(ocr_text):
    data = {}
    
    cn_match = re.search(r"C/N No\.\s*:\s*(\S+)", ocr_text)
    if cn_match:
        data["Consignment Note Number"] = cn_match.group(1)
    
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
    
    def re_search(pattern, text):
        match = re.search(pattern, text)
        return match.group(1).strip() if match else None

    data["Packages"] = {
        "No. of Packages": re_search(r"No\. of Packages:\s*(\S+)", ocr_text),
        "Gross Weight": re_search(r"Gross Wt\.\s*(\d+)", ocr_text),
        "Chargeable Weight": re_search(r"Chargeable Wt\.\s*(\d+)", ocr_text)
    }
    
    data["Tracking"] = {
        "Booking No.": re_search(r"Booking No\.\s*:\s*(\S+)", ocr_text),
        "Vehicle No.": re_search(r"Truck No\.\s*:\s*(\S+)", ocr_text)
    }
    
    freight_match = re.search(r"Freight Charge:\s*([\d.,]+)", ocr_text)
    if freight_match:
        data["Freight Charge"] = freight_match.group(1)
    
    return data

def process_pdf_with_easyocr(pdf_path):
    reader = easyocr.Reader(['en'])
    images = convert_from_path(pdf_path, dpi=300)
    results = []
    for idx, image in enumerate(images):
        print(f"Processing page {idx+1} with EasyOCR...")
        img_np = np.array(image)
        result = reader.readtext(img_np, detail=0, paragraph=True)
        ocr_text = "\n".join(result)
        print("Raw OCR Text:")
        print(ocr_text)
        structured_data = extract_structured_data(ocr_text)
        print("Structured Data:")
        print(json.dumps(structured_data, indent=2))
        results.append(structured_data)
    return results

PDF_PATH = "truck consignment.pdf"
process_pdf_with_easyocr(PDF_PATH)
