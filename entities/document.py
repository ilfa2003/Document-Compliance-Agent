from typing import Any, Dict, List, Optional

class Document:
    def __init__(self, doc_id: str, doc_type: str, data: Dict[str, Any], embedding: Optional[List[float]] = None):
        self.doc_id = doc_id
        self.doc_type = doc_type
        self.data = data
        self.embedding = embedding 