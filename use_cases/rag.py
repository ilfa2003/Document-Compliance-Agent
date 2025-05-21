from adapters.mongo_repository import MongoRepository
from adapters.llm_service import LLMService
from entities.document import Document
from use_cases.compliance import RULES
from entities.result import ComplianceResult
from typing import Dict, List
import json
from use_cases.compliance_rules import USER_RULES

def vector_search_similar_docs(query_embedding, top_k=3) -> List[dict]:
    repo = MongoRepository()
    collection = repo.collection
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
    return similar_docs

def generate_rag_compliance_report(links: Dict[str, Document], deterministic_results: List[ComplianceResult]) -> str:
    # For each main doc, retrieve similar docs using its embedding
    rag_context = {}
    for key, doc in links.items():
        if doc and doc.embedding:
            similar = vector_search_similar_docs(doc.embedding, top_k=3)
            rag_context[key] = similar
    llm = LLMService()
    prompt_template = """
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
    context = {
        "rules": USER_RULES,
        "documents": json.dumps({k: v.data if v else None for k, v in links.items()}, indent=2, default=str),
        "deterministic_results": json.dumps([
            {"label": r.rule_name, "passed": r.passed, "explanation": r.explanation} for r in deterministic_results
        ], indent=2, ensure_ascii=False),
        "rag_context": json.dumps(rag_context, indent=2, default=str)
    }
    return llm.generate_report(prompt_template, context)

def normalize_value(val):
    if not val:
        return ""
    return str(val).replace("-", "").replace(" ", "").strip().lower()

def get_first_present(data, keys):
    for key in keys:
        if key in data:
            return data[key]
    return None

def link_documents(docs: List[Document]) -> Dict[str, Document]:
    links = {
        "invoice": None,
        "customs_declaration": None,
        "waybill": None,
        "customs_certificate": None
    }

    # Step 1: Find the invoice document
    for doc in docs:
        data = doc.data
        invoice_no = get_first_present(data, ["Invoice number", "invoice_number", "Invoice No."])
        if invoice_no:
            links["invoice"] = doc
            break  # Assuming only one invoice per batch

    # Step 2: Extract target values from the invoice
    target_invoice_no = normalize_value(get_first_present(links["invoice"].data, ["Invoice number", "invoice_number", "Invoice No."])) if links["invoice"] else ""
    target_dec_no = normalize_value(get_first_present(links["invoice"].data, ["Declaration Number (DEC NO.)", "declaration_number", "DEC NO.", "Declaration No."])) if links["invoice"] else ""
    target_crn_no = normalize_value(get_first_present(links["invoice"].data, ["CRN No.", "crn_no", "CRN Number"])) if links["invoice"] else ""
    target_bill_no = normalize_value(get_first_present(links["invoice"].data, ["bill_number", "Bill Number", "Bill No."])) if links["invoice"] else ""

    # Step 3: Match other documents using the extracted values
    for doc in docs:
        data = doc.data
        # Customs Declaration
        dec_no = get_first_present(data, ["Declaration Number (DEC NO.)", "declaration_number", "DEC NO.", "Declaration No."])
        if normalize_value(dec_no) == target_dec_no and doc != links["invoice"]:
            links["customs_declaration"] = doc
        # Waybill
        crn_no = get_first_present(data, ["CRN No.", "crn_no", "CRN Number"])
        if normalize_value(crn_no) == target_crn_no and doc != links["invoice"]:
            links["waybill"] = doc
        # Customs Certificate
        bill_no = get_first_present(data, ["bill_number", "Bill Number", "Bill No."])
        if normalize_value(bill_no) == target_bill_no and doc != links["invoice"]:
            links["customs_certificate"] = doc

    return links 