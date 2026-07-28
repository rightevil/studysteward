"""
AI provider system — preset providers with OpenAI-compatible API.
User just picks a provider name and provides an API key.
"""
import json
import httpx
from dataclasses import dataclass
from ai.base import AIBackend, Summary, Chunk

# Preset providers. Each has a base_url and default model.
# All use OpenAI-compatible chat completions API.
PRESETS: dict[str, dict] = {
    "claude": {
        "base_url": "https://api.anthropic.com/v1/messages",
        "model": "claude-sonnet-4-6",
        "protocol": "anthropic",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o",
        "protocol": "openai",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-chat",
        "protocol": "openai",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1/chat/completions",
        "model": "moonshot-v1-8k",
        "protocol": "openai",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        "model": "glm-4-flash",
        "protocol": "openai",
    },
    "agnes": {
        "base_url": "https://apihub.agnes-ai.com/v1/chat/completions",
        "model": "agnes-2.0-flash",
        "protocol": "openai",
    },
    "custom": {
        "base_url": "",
        "model": "",
        "protocol": "openai",
    },
}

SYSTEM_PROMPT = "You are a helpful learning assistant. Respond concisely and accurately."

SUMMARIZE_PROMPT = """Analyze the following document text and return a JSON object with:
- title: a concise descriptive title
- keywords: 3-5 key topic keywords as a list
- summary: a one-paragraph summary (2-3 sentences)

Return ONLY the JSON object, no other text.

Document text:
{text}"""

ASK_PROMPT = """Answer the question based on the provided context chunks AND your own knowledge.
- Use [doc_title] markers when citing from the provided context.
- If the context is relevant, prioritize it. If not, use your own knowledge freely.
- Clearly distinguish what comes from the documents vs. your general knowledge.

Context:
{context}

Question: {question}"""

TAG_PROMPT = """Suggest 2-5 hierarchical tags for this document content. Tags should follow a "Category/Subcategory" format.
Return one tag per line, no other text.

Content:
{text}"""


class AIProvider(AIBackend):
    """Single backend that works with any OpenAI-compatible or Anthropic API."""

    def __init__(self, provider: str, api_key: str, base_url: str = "",
                 model: str = "", protocol: str = "openai"):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.protocol = protocol

    def _call_openai(self, prompt: str, max_tokens: int = 1024, stream: bool = False):
        """Call OpenAI-compatible API. Returns str if stream=False, generator if stream=True."""
        client = httpx.Client(timeout=120)
        if stream:
            req = client.build_request(
                "POST", self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": True,
                },
            )
            resp = client.send(req, stream=True)

            def generate():
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        import json
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices") or [{}]
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                client.close()
            return generate()
        else:
            try:
                resp = client.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            finally:
                client.close()

    def _call_anthropic(self, prompt: str, max_tokens: int = 1024, stream: bool = False):
        """Call Anthropic API."""
        with httpx.Client(timeout=120) as client:
            if stream:
                req = client.build_request(
                    "POST", self.base_url,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                    },
                )
                resp = client.send(req, stream=True)

                def generate():
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            import json
                            event = json.loads(line[6:])
                            if event.get("type") == "content_block_delta":
                                text = event.get("delta", {}).get("text", "")
                                if text:
                                    yield text
                return generate()
            else:
                resp = client.post(
                    self.base_url,
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": max_tokens,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                return data["content"][0]["text"]

    def _call(self, prompt: str, max_tokens: int = 1024, stream: bool = False):
        if self.protocol == "anthropic":
            return self._call_anthropic(prompt, max_tokens, stream)
        return self._call_openai(prompt, max_tokens, stream)

    def summarize(self, text: str) -> Summary:
        response = self._call(SUMMARIZE_PROMPT.format(text=text[:8000]))
        data = json.loads(response)
        return Summary(
            title=data.get("title", "Untitled"),
            keywords=data.get("keywords", []),
            summary=data.get("summary", ""),
        )

    def ask(self, question: str, context: list[Chunk]) -> str:
        ctx_parts = [f"[{c.doc_title}]\n{c.content}" for c in context]
        ctx_text = "\n\n".join(ctx_parts)
        return self._call(ASK_PROMPT.format(context=ctx_text, question=question))

    def ask_stream(self, question: str, context: list[Chunk]):
        """Stream tokens for Q&A. Yields text chunks."""
        ctx_parts = [f"[{c.doc_title}]\n{c.content}" for c in context]
        ctx_text = "\n\n".join(ctx_parts)
        return self._call(ASK_PROMPT.format(context=ctx_text, question=question), stream=True)

    def suggest_tags(self, text: str) -> list[str]:
        response = self._call(TAG_PROMPT.format(text=text[:4000]))
        return [line.strip() for line in response.strip().split("\n") if line.strip()]


def create_provider_from_env(config: dict) -> AIProvider:
    """Create an AI provider from configuration dict (from .env)."""
    name = config.get("provider", "claude")
    api_key = config.get("api_key", "")
    preset = PRESETS.get(name, PRESETS["custom"])

    base_url = config.get("base_url") or preset["base_url"]
    model = config.get("model") or preset["model"]
    protocol = preset["protocol"]

    if name == "custom" and (not base_url or not model):
        raise ValueError("Custom provider requires base_url and model in config")

    return AIProvider(
        provider=name,
        api_key=api_key,
        base_url=base_url,
        model=model,
        protocol=protocol,
    )
