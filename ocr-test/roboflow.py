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
API_KEY = os.getenv("ROBOFLOW_API_KEY")  # Ensure this environment variable is set

# Path to your PDF file
PDF_PATH = "invoice (1).PDF"

# Convert PDF pages to images
images = convert_from_path(PDF_PATH, dpi=300)

# Function to convert PIL image to base64
def pil_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Function to extract structured data from OCR result
def extract_structured_data(ocr_text):
    data = {}
    # Example regex patterns; adjust based on your document structure
    # Invoice Number (pattern: Invoice No [whitespace or colon] VALUE)
    invoice_no_match = re.search(r"Invoice\s+No[\s:]*([A-Z0-9\-]+)", ocr_text, re.IGNORECASE)
    if invoice_no_match:
        data["Invoice Number"] = invoice_no_match.group(1)

    # Invoice Date (pattern: Invoice Date [whitespace or colon] VALUE)
    invoice_date_match = re.search(r"Invoice\s+Date[\s:]*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})", ocr_text)
    if invoice_date_match:
        data["Invoice Date"] = invoice_date_match.group(1)

    # Total Amount (Including VAT)
    total_amount_match = re.search(r"Total Amount.*?Including.*?VAT\W*[:]*\s*([\d.,]+)", ocr_text)
    if total_amount_match:
        data["Total Amount (Including VAT)"] = total_amount_match.group(1)

    # VAT Amount
    vat_amount_match = re.search(r"VAT Amount\s*[:]*\s*([\d.,]+)", ocr_text)
    if vat_amount_match:
        data["VAT Amount"] = vat_amount_match.group(1)

    # Customer Name
    customer_name_match = re.search(r"Customer\s+Name\s*:?(.+?)Invoice Date", ocr_text, re.DOTALL)
    if customer_name_match:
        data["Customer Name"] = customer_name_match.group(1).strip()

    return data

# Process each image
for idx, image in enumerate(images):
    print(f"Processing page {idx + 1}...")

    # Convert image to base64
    image_base64 = pil_to_base64(image)

    # Prepare payload for OCR API
    payload = {
        "image": {
            "type": "base64",
            "value": image_base64
        }
    }

    # Send request to Roboflow OCR API
    response = requests.post(
        f"https://infer.roboflow.com/doctr/ocr?api_key={API_KEY}",
        json=payload
    )

    if response.status_code == 200:
        ocr_result = response.json()
        ocr_text = ocr_result.get("result", "")
        print("OCR Text:")
        print(ocr_text)

        # Extract structured data
        structured_data = extract_structured_data(ocr_text)
        print("Extracted Structured Data:")
        print(json.dumps(structured_data, indent=2))
    else:
        print(f"Failed to process page {idx + 1}. Status Code: {response.status_code}")
        print(response.text)
