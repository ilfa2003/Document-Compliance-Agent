from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

class LLMService:
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.llm = GoogleGenerativeAI(model=model_name, google_api_key=self.api_key)

    def generate_report(self, prompt_template: str, context: dict) -> str:
        prompt = PromptTemplate(
            input_variables=list(context.keys()),
            template=prompt_template
        )
        chain = prompt | self.llm
        return chain.invoke(context) 