from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        # Expects OPENAI_API_KEY in environment
        api_key = os.getenv("OPENAI_API_KEY")
        self.mock_mode = False
        if not api_key:
            print("Warning: OPENAI_API_KEY not found. Using Mock Mode.")
            self.mock_mode = True
        else:
            self.client = OpenAI(api_key=api_key)

    def generate_response(self, text: str, category: str, urgency: str, snippets: list) -> dict:
        if self.mock_mode:
            return {
                "action_plan": ["Mock Step 1: Check System", "Mock Step 2: Inform Customer"],
                "customer_reply_draft": f"Dear Customer, we received your {category} complaint (Urgency: {urgency}). We are working on it. (MOCK RESPONSE)",
                "risk_flags": ["MOCK_MODE_ACTIVE"]
            }

        context = "\n".join(snippets)
        prompt = f"""
        You are a helpful banking customer support assistant.
        Valid Categories: {category}
        Urgency: {urgency}
        
        Relevant Procedures (SOPs):
        {context}
        
        Customer Complaint:
        {text}
        
        Task:
        1. Create a step-by-step action plan for the agent.
        2. Draft a polite, professional response to the customer in Turkish.
        3. Identify any risk flags (PII leak, legal threat, etc.).
        
        Output JSON Format:
        {{
            "action_plan": ["step 1", "step 2"],
            "customer_reply_draft": "string",
            "risk_flags": ["flag1"]
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", # or gpt-4
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that outputs JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            import json
            content = response.choices[0].message.content
            # Clean possible markdown code fences
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            return json.loads(content)
        except Exception as e:
            print(f"LLM Error: {e}")
            return {
                "action_plan": ["Error calling LLM"],
                "customer_reply_draft": "System Error: Could not generate draft.",
                "risk_flags": ["LLM_ERROR"]
            }

llm_client = LLMClient()
