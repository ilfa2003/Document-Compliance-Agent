import os
import pymongo
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
if not GOOGLE_API_KEY or not MONGO_URI:
    raise ValueError("Missing GOOGLE_API_KEY or MONGO_URI in environment variables.")

# 2. Connect to MongoDB and fetch all relevant documents
client = pymongo.MongoClient(MONGO_URI)
db = client["document_compliance"]  # Change if your DB name is different
collection = db["invoices"]         # Change if your collection name is different

docs = list(collection.find({}))  # You can filter by case, date, etc.

# 3. Link documents by key references
def link_documents(docs):
    links = {
        "invoice": None,
        "customs_declaration": None,
        "waybill": None,
        "customs_certificate": None
    }
    for doc in docs:
        if doc.get("Invoice number") == "LACO-39":
            links["invoice"] = doc
        if doc.get("Declaration Number (DEC NO.)") == "203-04144376-23":
            links["customs_declaration"] = doc
        if doc.get("CRN No.") == "319303":
            links["waybill"] = doc
        if doc.get("bill_number") == "203-04144376-23":
            links["customs_certificate"] = doc
    return links

linked_docs = link_documents(docs)

# 4. Prepare user rules (natural language)
user_rules = """
Match invoice LACO-39 to customs declaration 203-04144376-23 and consignment note 319303 using the respective fields.
Ensure invoice date is on or after customs certificate date.
Check total weight matches across all documents.
Check exporter name consistency.
Check vehicle/container numbers match.
Generate a compliance report with pass/fail for each rule and explanations.
"""

# 5. Deterministic validation for each rule
def check_invoice_to_customs_match(links):
    if links["invoice"] and links["customs_declaration"]:
        return True, f"DEC Matched: {links['customs_declaration'].get('Declaration Number (DEC NO.)')}"
    return False, "DEC not matched."

def check_invoice_to_waybill_match(links):
    if links["invoice"] and links["waybill"]:
        return True, f"CRN No. Matched: {links['waybill'].get('CRN No.')}"
    return False, "CRN No. not matched."

def check_total_weight(links):
    invoice_weight = links["invoice"].get("Total weight") if links["invoice"] else None
    customs_weight = links["customs_declaration"].get("Gross Weight") if links["customs_declaration"] else None
    if invoice_weight and customs_weight:
        if str(invoice_weight).strip() == str(customs_weight).strip().replace(' kg', '').replace('KG', ''):
            return True, f"Total Weight Matches ({invoice_weight})"
        else:
            return False, f"Weight Mismatch: Invoice: {invoice_weight}, Customs: {customs_weight}"
    return False, "Weight data missing in one or more documents."

def check_exporter_name(links):
    invoice_exporter = links["invoice"].get("Shipper/Exporter details", {}).get("company_name") if links["invoice"] else None
    customs_exporter = links["customs_declaration"].get("Consignee/Exporter") if links["customs_declaration"] else None
    if invoice_exporter and customs_exporter:
        if invoice_exporter.lower() in customs_exporter.lower():
            return True, "Exporter Name Consistent"
        else:
            return False, f"Exporter Name Mismatch: Invoice: {invoice_exporter}, Customs: {customs_exporter}"
    return False, "Exporter name missing in one or more documents."

def check_vehicle_number(links):
    invoice_vehicle = None
    if links["invoice"]:
        for ref in links["invoice"].get("LAC reference numbers", []):
            if isinstance(ref, str) and "DXB" in ref:
                invoice_vehicle = ref
    customs_vehicle = links["customs_certificate"].get("container_vehicle_number") if links["customs_certificate"] else None
    if invoice_vehicle and customs_vehicle:
        if invoice_vehicle == customs_vehicle:
            return True, f"Vehicle/Container Number Matches ({invoice_vehicle})"
        else:
            return False, f"Vehicle/Container Number Mismatch: Invoice: {invoice_vehicle}, Certificate: {customs_vehicle}"
    return False, "Vehicle/container number missing in one or more documents."

def check_invoice_date_vs_certificate(links):
    invoice_date = links["invoice"].get("Invoice date") if links["invoice"] else None
    cert_date = links["customs_certificate"].get("invoice_date") if links["customs_certificate"] else None
    if invoice_date and cert_date:
        if invoice_date >= cert_date:
            return True, "Invoice Date ≥ Customs Certificate Date"
        else:
            return False, f"Invoice Date {invoice_date} < Certificate Date {cert_date}"
    return False, "Invoice or certificate date missing."

rule_validators = [
    ("DEC Matched", check_invoice_to_customs_match),
    ("CRN No. Matched", check_invoice_to_waybill_match),
    ("Total Weight Match", check_total_weight),
    ("Exporter Name Consistency", check_exporter_name),
    ("Vehicle/Container Number Match", check_vehicle_number),
    ("Invoice Date vs Certificate", check_invoice_date_vs_certificate),
]

deterministic_results = []
for label, func in rule_validators:
    passed, explanation = func(linked_docs)
    deterministic_results.append({
        "label": label,
        "passed": passed,
        "explanation": explanation
    })

# 6. Use Gemini LLM to reason and generate the compliance report, with deterministic results as context
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

prompt = PromptTemplate(
    input_variables=["rules", "documents", "deterministic_results"],
    template="""
You are a compliance officer. Given these compliance rules:
{rules}

And these linked business documents in JSON:
{documents}

Here are the results of deterministic (programmatic) validation for each rule:
{deterministic_results}

For each rule, output a checklist with pass/fail and a concise explanation, using the deterministic results as your primary guide. If the deterministic result is ambiguous or missing, use your own reasoning. At the end, summarize the overall compliance status in the style:

Invoice_INV1001 – Compliance Report

✅ PO Matched: PO505
✅ GRN Matched: GRN802
❌ Quantity Mismatch: Item B (Invoice: 3 > GRN: 2)
❌ Unit Price Mismatch: Item B ($25 vs $20 in PO)
✅ Total Amount Within Tolerance
✅ Invoice Date ≥ GRN Date
✅ Vendor Approved (TechSupply Inc.)

Final Status: FAIL (2 issues found)
"""
)

chain = prompt | llm

report = chain.invoke({
    "rules": user_rules,
    "documents": json.dumps(linked_docs, indent=2, default=str),
    "deterministic_results": json.dumps(deterministic_results, indent=2, ensure_ascii=False)
})

print(report)
