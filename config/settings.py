import os
from dotenv import load_dotenv

load_dotenv()

#Make sure to add the correct environment variables to the .env file
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_NAME = "document_compliance" #You can change the database name to any other name.