from entities.document import Document
from typing import List, Dict, Any, Optional
from fallbacks.extract_invoice2 import extract_leminar_invoice_data
from fallbacks.extract_invoice3 import extract_western_express_data
from fallbacks.extract_invoice4 import extract_customs_certificate_data
from fallbacks.extract_invoice5 import extract_customs_declaration_data
import os

EXTRACTION_MAP = {
    'leminar_invoice': extract_leminar_invoice_data,
    'western_express': extract_western_express_data,
    'customs_certificate': extract_customs_certificate_data,
    'customs_declaration': extract_customs_declaration_data,
}

def detect_document_type(filename: str) -> Optional[str]:
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

def batch_extract(files: List[str], doc_type: str = None, add_embedding: bool = True) -> List[Document]:
    results = []
    for file_path in files:
        if not os.path.isfile(file_path):
            continue
        dtype = doc_type or detect_document_type(file_path)
        if dtype not in EXTRACTION_MAP:
            continue
        print(f"Extracting {file_path} as {dtype}...")

        # Use the process_and_save_* function for each type
        if dtype == 'leminar_invoice':
            from fallbacks.extract_invoice2 import process_and_save_leminar_invoice
            result = process_and_save_leminar_invoice(file_path, add_embedding=add_embedding)
            data = result.get("data", {})
        elif dtype == 'western_express':
            from fallbacks.extract_invoice3 import process_and_save_western_express
            result = process_and_save_western_express(file_path, add_embedding=add_embedding)
            data = result.get("data", {})
        elif dtype == 'customs_certificate':
            from fallbacks.extract_invoice4 import process_and_save_customs_certificate
            result = process_and_save_customs_certificate(file_path, add_embedding=add_embedding)
            data = result.get("data", {})
        elif dtype == 'customs_declaration':
            from fallbacks.extract_invoice5 import process_and_save_customs_declaration
            result = process_and_save_customs_declaration(file_path, add_embedding=add_embedding)
            data = result.get("data", {})
        else:
            data = EXTRACTION_MAP[dtype](file_path)

        doc = Document(doc_id=file_path, doc_type=dtype, data=data)
        results.append(doc)
    return results 