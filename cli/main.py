import argparse
from use_cases.extract import batch_extract
from adapters.mongo_repository import MongoRepository
from adapters.llm_service import LLMService
from adapters.file_adapter import FileAdapter
from use_cases.compliance import link_documents, run_deterministic_checks, generate_compliance_report
from use_cases.rag import generate_rag_compliance_report

# For demonstration, compliance logic is now wired up

def main():
    parser = argparse.ArgumentParser(description='Document Compliance Agent CLI (Clean Architecture)')
    parser.add_argument('inputs', nargs='+', help='PDF files or a directory containing PDFs')
    parser.add_argument('--type', help='Specify document type for all files (optional)')
    parser.add_argument('--extract', action='store_true', help='Extract and save documents to MongoDB')
    parser.add_argument('--report', action='store_true', help='Generate compliance report')
    parser.add_argument('--rag_report', action='store_true', help='Generate RAG+CAG compliance report')
    args = parser.parse_args()

    # Gather all PDF files
    input_files = []
    for inp in args.inputs:
        import os
        if os.path.isdir(inp):
            input_files.extend([os.path.join(inp, f) for f in os.listdir(inp) if f.lower().endswith('.pdf')])
        elif os.path.isfile(inp) and inp.lower().endswith('.pdf'):
            input_files.append(inp)

    if not input_files and args.extract:
        print('No PDF files found to process.')
        return

    if args.extract:
        docs = batch_extract(input_files, doc_type=args.type)
        repo = MongoRepository()
        for doc in docs:
            repo.save_document(doc)
        print(f"Extracted and saved {len(docs)} documents to MongoDB.")

    if args.report:
        repo = MongoRepository()
        docs = repo.get_documents()
        links = link_documents(docs)
        deterministic_results = run_deterministic_checks(links)
        report = generate_compliance_report(links, deterministic_results)
        print('\n' + '='*60)
        print('COMPLIANCE REPORT')
        print('='*60)
        print(report)
        print('='*60 + '\n')
        FileAdapter.save_text('compliance_report.txt', report)
        print('Compliance report saved to compliance_report.txt')

    if args.rag_report:
        repo = MongoRepository()
        docs = repo.get_documents()
        links = link_documents(docs)
        deterministic_results = run_deterministic_checks(links)
        report_rag = generate_rag_compliance_report(links, deterministic_results)
        print('\n' + '='*60)
        print('RAG+CAG COMPLIANCE REPORT')
        print('='*60)
        print(report_rag)
        print('='*60 + '\n')
        FileAdapter.save_text('compliance_report_rag.txt', report_rag)
        print('RAG+CAG compliance report saved to compliance_report_rag.txt')

if __name__ == '__main__':
    main() 