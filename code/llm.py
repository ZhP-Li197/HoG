import json
import os
import re
import time

from openai import OpenAI


class LLM:
    def __init__(self, model, api_key=None, base_url=None, timeout=180, retries=3):
        key = api_key or os.environ.get("HOG_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set HOG_API_KEY or OPENAI_API_KEY")
        options = {"api_key": key, "timeout": timeout, "max_retries": 0}
        address = base_url or os.environ.get("HOG_BASE_URL")
        if address:
            options["base_url"] = address
        self.client = OpenAI(**options)
        self.model = model
        self.retries = retries

    def json(self, instruction, payload):
        messages = [
            {"role": "system", "content": instruction + " Return one valid JSON object only."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        error = None
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0)
                content = response.choices[0].message.content or ""
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if not match:
                    raise ValueError(f"No JSON object in model response: {content[:300]}")
                result = json.loads(match.group(0))
                if not isinstance(result, dict):
                    raise ValueError("Model response must be a JSON object")
                return result
            except Exception as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM request failed: {error}")
