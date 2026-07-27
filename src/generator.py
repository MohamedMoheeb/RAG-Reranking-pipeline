import os
from typing import List, Dict, Any
from google import genai


class GeminiGenerator:
    """Uses Google Gemini to generate grounded answers based on retrieved context."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY environment variable not set.")
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model_name = model_name

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Constructs a grounded RAG prompt and calls the Gemini API."""
        combined_context = "\n\n".join(
            [f"[Context {i+1}]: {item['text']}" for i, item in enumerate(context_chunks)]
        )

        prompt = f"""
You are an expert AI assistant. Answer the user's question accurately and concisely using ONLY the provided contexts below.
If the context does not contain enough information to answer the question, state that clearly without making up facts.

--- RETRIEVED CONTEXT ---
{combined_context}

--- USER QUESTION ---
{query}

--- ANSWER ---
"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        return response.text
