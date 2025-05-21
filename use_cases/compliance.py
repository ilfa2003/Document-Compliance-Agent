from entities.document import Document
from entities.compliance_rule import ComplianceRule
from entities.result import ComplianceResult
from adapters.llm_service import LLMService
from typing import List, Dict, Any
from use_cases.compliance_rules import USER_RULES

class ComplianceChecker:
    def __init__(self, rules: List[ComplianceRule]):
        self.rules = rules

    def check(self, docs: List[Document]) -> List[ComplianceResult]:
        results = []
        for rule in self.rules:
            passed, explanation = rule.validate(docs)
            results.append(ComplianceResult(rule.name, passed, explanation))
        return results 

# --- Robust linking logic for main compliance case ---
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

# --- Deterministic rule functions ---
def check_invoice_to_customs_match(links):
    if links["invoice"] and links["customs_declaration"]:
        return True, f"DEC Matched: {get_first_present(links['customs_declaration'].data, ['Declaration Number (DEC NO.)', 'declaration_number', 'DEC NO.', 'Declaration No.'])}"
    return False, "DEC not matched."

def check_invoice_to_waybill_match(links):
    if links["invoice"] and links["waybill"]:
        return True, f"CRN No. Matched: {get_first_present(links['waybill'].data, ['CRN No.', 'crn_no', 'CRN Number'])}"
    return False, "CRN No. not matched."

def check_total_weight(links):
    invoice_weight = get_first_present(links["invoice"].data, ["Total weight", "total_weight"]) if links["invoice"] else None
    customs_weight = get_first_present(links["customs_declaration"].data, ["Gross Weight", "gross_weight"]) if links["customs_declaration"] else None
    if invoice_weight and customs_weight:
        if normalize_value(invoice_weight) == normalize_value(customs_weight):
            return True, f"Total Weight Matches ({invoice_weight})"
        else:
            return False, f"Weight Mismatch: Invoice: {invoice_weight}, Customs: {customs_weight}"
    return False, "Weight data missing in one or more documents."

def check_exporter_name(links):
    invoice_exporter = get_first_present(links["invoice"].data.get("Shipper/Exporter details", {}), ["company_name", "Company Name"]) if links["invoice"] else None
    customs_exporter = get_first_present(links["customs_declaration"].data, ["Consignee/Exporter", "consignee_exporter"]) if links["customs_declaration"] else None
    if invoice_exporter and customs_exporter:
        if normalize_value(invoice_exporter) in normalize_value(customs_exporter):
            return True, "Exporter Name Consistent"
        else:
            return False, f"Exporter Name Mismatch: Invoice: {invoice_exporter}, Customs: {customs_exporter}"
    return False, "Exporter name missing in one or more documents."

def check_vehicle_number(links):
    invoice_vehicle = None
    if links["invoice"]:
        for ref in links["invoice"].data.get("LAC reference numbers", []):
            if isinstance(ref, str) and "DXB" in ref:
                invoice_vehicle = ref
    customs_vehicle = get_first_present(links["customs_certificate"].data, ["container_vehicle_number", "Container/Vehicle Number"]) if links["customs_certificate"] else None
    if invoice_vehicle and customs_vehicle:
        if normalize_value(invoice_vehicle) == normalize_value(customs_vehicle):
            return True, f"Vehicle/Container Number Matches ({invoice_vehicle})"
        else:
            return False, f"Vehicle/Container Number Mismatch: Invoice: {invoice_vehicle}, Certificate: {customs_vehicle}"
    return False, "Vehicle/container number missing in one or more documents."

def check_invoice_date_vs_certificate(links):
    invoice_date = get_first_present(links["invoice"].data, ["Invoice date", "invoice_date"]) if links["invoice"] else None
    cert_date = get_first_present(links["customs_certificate"].data, ["invoice_date", "Invoice date"]) if links["customs_certificate"] else None
    if invoice_date and cert_date:
        if invoice_date >= cert_date:
            return True, "Invoice Date ≥ Customs Certificate Date"
        else:
            return False, f"Invoice Date {invoice_date} < Certificate Date {cert_date}"
    return False, "Invoice or certificate date missing."

RULES = [
    ("DEC Matched", check_invoice_to_customs_match),
    ("CRN No. Matched", check_invoice_to_waybill_match),
    ("Total Weight Match", check_total_weight),
    ("Exporter Name Consistency", check_exporter_name),
    ("Vehicle/Container Number Match", check_vehicle_number),
    ("Invoice Date vs Certificate", check_invoice_date_vs_certificate),
]

def run_deterministic_checks(links) -> List[ComplianceResult]:
    results = []
    for label, func in RULES:
        passed, explanation = func(links)
        results.append(ComplianceResult(label, passed, explanation))
    return results

# --- LLM-based report generation ---
def generate_compliance_report(links, deterministic_results) -> str:
    llm = LLMService()
    prompt_template = """
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
    import json
    context = {
        "rules": USER_RULES,
        "documents": json.dumps({k: v.data if v else None for k, v in links.items()}, indent=2, default=str),
        "deterministic_results": json.dumps([
            {"label": r.rule_name, "passed": r.passed, "explanation": r.explanation} for r in deterministic_results
        ], indent=2, ensure_ascii=False)
    }
    return llm.generate_report(prompt_template, context) 