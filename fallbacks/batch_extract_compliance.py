import os
import argparse
from typing import List, Dict, Any
from bson import ObjectId
import json

# Import all extraction and saving functions from the four scripts
from extract_invoice2 import extract_leminar_invoice_data, process_and_save_leminar_invoice
from extract_invoice3 import extract_western_express_data, process_and_save_western_express
from extract_invoice4 import extract_customs_certificate_data, process_and_save_customs_certificate
from extract_invoice5 import extract_customs_declaration_data, process_and_save_customs_declaration

# Map document types to their processing functions
DOCUMENT_TYPE_MAP = {
    'leminar_invoice': process_and_save_leminar_invoice,
    'western_express': process_and_save_western_express,
    'customs_certificate': process_and_save_customs_certificate,
    'customs_declaration': process_and_save_customs_declaration,
}

# Simple filename-based type detection (can be improved)
def detect_document_type(filename: str) -> str:
    name = filename.lower()
    if 'leminar' in name or 'invoice' in name:
        return 'leminar_invoice'
    if 'waybill' in name or 'western' in name or 'consignment' in name:
        return 'western_express'
    if 'customs-certificate' in name or 'entryexit' in name:
        return 'customs_certificate'
    if 'customs' in name or 'declaration' in name:
        return 'customs_declaration'
    return None

def batch_process(files: List[str], doc_type: str = None, add_embedding: bool = False) -> List[Dict[str, Any]]:
    results = []
    for file_path in files:
        if not os.path.isfile(file_path):
            results.append({'file': file_path, 'status': 'error', 'message': 'File not found'})
            continue
        dtype = doc_type or detect_document_type(file_path)
        if dtype not in DOCUMENT_TYPE_MAP:
            results.append({'file': file_path, 'status': 'error', 'message': f'Unknown document type: {dtype}'})
            continue
        print(f"Processing {file_path} as {dtype}...")
        try:
            result = DOCUMENT_TYPE_MAP[dtype](file_path, add_embedding=add_embedding)
            result['file'] = file_path
            results.append(result)
        except Exception as e:
            results.append({'file': file_path, 'status': 'error', 'message': str(e)})
    return results

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj

# Compliance agent imports
import pymongo
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Compliance Agent Functions ---
def fetch_documents_from_mongo():
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError("Missing MONGO_URI in environment variables.")
    client = pymongo.MongoClient(MONGO_URI)
    db = client["document_compliance"] #You can change the database name to any other name.
    collection = db["invoices-test"] #You can change the collection name to any other name.
    docs = list(collection.find({}))
    client.close()
    return docs

def link_documents(docs):
    # This can be made more flexible
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

# Deterministic rule checks 
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

def generate_compliance_report(user_rules, linked_docs, deterministic_results):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
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
    return report

# --- END Compliance Agent Functions ---

def vector_search_similar_docs(query_embedding, top_k=3):
    """Retrieve top-k similar documents from MongoDB using vector search on the embedding field."""
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError("Missing MONGO_URI in environment variables.")
    client = pymongo.MongoClient(MONGO_URI)
    db = client["document_compliance"] #You can change the database name to any other name.
    collection = db["invoices-test"] #You can change the collection name to any other name.
    # MongoDB vector search pipeline (requires MongoDB Atlas vector search)
    pipeline = [
        {
            "$vectorSearch": {
                "index": "invoice_embedding_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 100,
                "limit": top_k
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
    try:
        similar_docs = list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"Vector search error: {e}")
        similar_docs = []
    client.close()
    return similar_docs

def main():
    parser = argparse.ArgumentParser(description='Batch process and extract data from various document PDFs.')
    parser.add_argument('inputs', nargs='+', help='PDF files or a directory containing PDFs')
    parser.add_argument('--type', choices=DOCUMENT_TYPE_MAP.keys(), help='Specify document type for all files (optional)')
    parser.add_argument('--embedding', action='store_true', help='Add Gemini embedding to MongoDB')
    parser.add_argument('--compliance', action='store_true', help='Run compliance agent after batch processing')
    parser.add_argument('--rag_compliance', action='store_true', help='Run RAG+CAG compliance agent after batch processing')
    args = parser.parse_args()

    # Gather all PDF files
    input_files = []
    for inp in args.inputs:
        if os.path.isdir(inp):
            input_files.extend([os.path.join(inp, f) for f in os.listdir(inp) if f.lower().endswith('.pdf')])
        elif os.path.isfile(inp) and inp.lower().endswith('.pdf'):
            input_files.append(inp)

    if not input_files:
        print('No PDF files found to process.')
        return

    results = batch_process(input_files, doc_type=args.type, add_embedding=args.embedding)

    # Print summary
    print('\nBatch Processing Summary:')
    for res in results:
        status = res.get('status', 'unknown')
        file = res.get('file', 'N/A')
        msg = res.get('message', '')
        print(f"- {file}: {status} {msg}")

    # Optionally, save summary to a JSON file
    with open('batch_processing_summary.json', 'w', encoding='utf-8') as f:
        import json
        serializable_results = make_json_serializable(results)
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    print('Summary saved to batch_processing_summary.json')

    # --- Compliance Agent Step ---
    if args.compliance:
        print('\nRunning Compliance Agent...')
        docs = fetch_documents_from_mongo()
        linked_docs = link_documents(docs)
        user_rules = """
Match invoice LACO-39 to customs declaration 203-04144376-23 and consignment note 319303 using the respective fields.
Ensure invoice date is on or after customs certificate date.
Check total weight matches across all documents.
Check exporter name consistency.
Check vehicle/container numbers match.
Generate a compliance report with pass/fail for each rule and explanations.
"""
        deterministic_results = []
        for label, func in rule_validators:
            passed, explanation = func(linked_docs)
            deterministic_results.append({
                "label": label,
                "passed": passed,
                "explanation": explanation
            })
        report = generate_compliance_report(user_rules, linked_docs, deterministic_results)
        print('\n' + '='*60)
        print('COMPLIANCE REPORT')
        print('='*60)
        print(report)
        print('='*60 + '\n')
        with open('compliance_report.txt', 'w', encoding='utf-8') as f:
            f.write(str(report))
        print('Compliance report saved to compliance_report.txt')

    # --- RAG+CAG Compliance Agent Step ---
    if args.rag_compliance:
        print('\nRunning RAG+CAG Compliance Agent...')
        docs = fetch_documents_from_mongo()
        linked_docs = link_documents(docs)
        user_rules = """
Match invoice LACO-39 to customs declaration 203-04144376-23 and consignment note 319303 using the respective fields.
Ensure invoice date is on or after customs certificate date.
Check total weight matches across all documents.
Check exporter name consistency.
Check vehicle/container numbers match.
Generate a compliance report with pass/fail for each rule and explanations.
Additionally, use any similar or related documents retrieved by vector search to inform your reasoning and catch edge cases or fuzzy matches.
"""
        deterministic_results = []
        for label, func in rule_validators:
            passed, explanation = func(linked_docs)
            deterministic_results.append({
                "label": label,
                "passed": passed,
                "explanation": explanation
            })
        # For each main doc, retrieve similar docs using its embedding
        rag_context = {}
        for key, doc in linked_docs.items():
            if doc and 'embedding' in doc:
                similar = vector_search_similar_docs(doc['embedding'], top_k=3)
                rag_context[key] = similar
        # Augment the prompt with RAG context
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)
        prompt = PromptTemplate(
            input_variables=["rules", "documents", "deterministic_results", "rag_context"],
            template="""
You are a compliance officer. Given these compliance rules:
{rules}

And these linked business documents in JSON:
{documents}

Here are the results of deterministic (programmatic) validation for each rule:
{deterministic_results}

Here are additional similar or related documents retrieved by vector search for each main document:
{rag_context}

For each rule, output a checklist with pass/fail and a concise explanation, using the deterministic results and retrieved context as your primary guide. If the deterministic result is ambiguous or missing, use your own reasoning. At the end, summarize the overall compliance status in the style:

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
        report_rag = chain.invoke({
            "rules": user_rules,
            "documents": json.dumps(linked_docs, indent=2, default=str),
            "deterministic_results": json.dumps(deterministic_results, indent=2, ensure_ascii=False),
            "rag_context": json.dumps(rag_context, indent=2, default=str)
        })
        print('\n' + '='*60)
        print('RAG+CAG COMPLIANCE REPORT')
        print('='*60)
        print(report_rag)
        print('='*60 + '\n')
        with open('compliance_report_rag.txt', 'w', encoding='utf-8') as f:
            f.write(str(report_rag))
        print('RAG+CAG compliance report saved to compliance_report_rag.txt')

if __name__ == '__main__':
    main() 