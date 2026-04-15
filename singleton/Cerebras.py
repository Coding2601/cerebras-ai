import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()

class LLM:
    _instance = None

    def __new__(cls, model_name: str = None):
        if cls._instance is None:
            cls._instance = super(LLM, cls).__new__(cls)
            cls._instance._initialize(model_name)
        return cls._instance

    def _initialize(self, model_name):
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("CEREBRAS_API_KEY not found in environment variables")
        self.client = Cerebras(api_key=api_key)
        self.model_name = model_name
    
    def complete(self, user_prompt, context=None, system_prompt=None, max_tokens=500, temperature=0.7):
        messages = []
        if system_prompt:
            messages.append({ "role": "system", "content": system_prompt })
        if context:
            context_text = "\n\n".join(context)
            user_prompt = f"""
                            Use the following Bhagavad Gita verses to answer the question.

                            Context:
                            {context_text}

                            Question:
                            {user_prompt}
                        """
        messages.append({ "role": "user", "content": user_prompt })
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM request failed: {str(e)}")