from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("complaintops.llm_client")

class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_plan: list[str] = Field(min_length=1)
    customer_reply_draft: str = Field(min_length=1)
    risk_flags: list[str] = Field(min_length=1)

class LLMClient:
    def __init__(self):
        # Expects OPENAI_API_KEY in environment
        api_key = os.getenv("OPENAI_API_KEY")
        self.mock_mode = False
        if not api_key:
            logger.warning("OPENAI_API_KEY not found. Using mock mode.")
            self.mock_mode = True
        else:
            self.client = OpenAI(api_key=api_key)

    def _build_prompt(self, text: str, category: str, urgency: str, snippets: list, strict_json: bool) -> str:
        context = "\n".join(snippets)
        json_instruction = (
            "Return ONLY valid JSON with double quotes and no markdown or code fences."
            if strict_json
            else "Output JSON Format:"
        )
        return f"""
        You are a helpful banking customer support assistant.
        Valid Categories: {category}
        Urgency: {urgency}
        
        Relevant Procedures (SOPs):
        {context}

        Sources (explicitly list in output as provided):
        {sources_context}
        
        Customer Complaint:
        {text}
        
        Task:
        1. Create a step-by-step action plan for the agent.
        2. Draft a polite, professional response to the customer in Turkish.
        3. Identify any risk flags (PII leak, legal threat, etc.).
        4. Include the sources array in the output.
        
        {json_instruction}
        {{
            "action_plan": ["step 1", "step 2"],
            "customer_reply_draft": "string",
            "risk_flags": ["flag1"],
            "sources": [
                {{
                    "doc_name": "string",
                    "source": "string",
                    "snippet": "string"
                }}
            ]
        }}
        """

    def _parse_and_validate(self, content: str) -> dict:
        cleaned = content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        validated = LLMResponse.model_validate(parsed)
        return validated.model_dump()

    def generate_response(self, text: str, category: str, urgency: str, snippets: list) -> dict:
        if self.mock_mode:
            return {
                "action_plan": ["Mock Step 1: Check System", "Mock Step 2: Inform Customer"],
                "customer_reply_draft": f"Dear Customer, we received your {category} complaint (Urgency: {urgency}). We are working on it. (MOCK RESPONSE)",
                "risk_flags": ["MOCK_MODE_ACTIVE"]
            }

        attempts = [
            self._build_prompt(text, category, urgency, snippets, strict_json=False),
            self._build_prompt(text, category, urgency, snippets, strict_json=True),
        ]

        for index, prompt in enumerate(attempts, start=1):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo", # or gpt-4
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant. Output only valid JSON, no markdown or code fences."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                )
                content = response.choices[0].message.content
                return self._parse_and_validate(content)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning("LLM JSON validation failed on attempt %s: %s", index, e)
                continue
            except Exception as e:
                logger.error("LLM Error on attempt %s: %s", index, e)
                break

        return {
            "action_plan": ["Error calling LLM"],
            "customer_reply_draft": "System Error: Could not generate draft.",
            "risk_flags": ["LLM_ERROR"]
        }

llm_client = LLMClient()
