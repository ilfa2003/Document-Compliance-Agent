from entities.document import Document
from typing import List, Dict, Any
import pymongo
import os
from dotenv import load_dotenv

#replace the collection name with the one you want to use
class MongoRepository:
    def __init__(self, collection_name: str = "invoices-test"):
        load_dotenv()
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = "document_compliance" #You can change the database name to any other name.
        self.collection_name = collection_name
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

    def get_documents(self, filter_dict: Dict[str, Any] = None) -> List[Document]:
        filter_dict = filter_dict or {}
        docs = []
        for doc in self.collection.find(filter_dict):
            docs.append(Document(
                doc_id=str(doc.get("_id")),
                doc_type=doc.get("document_type", "unknown"),
                data=doc,
                embedding=doc.get("embedding")
            ))
        return docs

    def save_document(self, doc: Document):
        data = doc.data.copy()
        if doc.embedding:
            data["embedding"] = doc.embedding
        result = self.collection.insert_one(data)
        return str(result.inserted_id) 