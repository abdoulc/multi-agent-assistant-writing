import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:1b",
        timeout: float = 180,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, system_prompt: str, context: dict[str, object], *, output_schema: dict[str, object] | None = None, ) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
        }
        if output_schema is not None:
            payload["format"] = output_schema
        request = Request(
            url=f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Ollama returned HTTP status {exc.code}.") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not connect to Ollama: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("The Ollama request timed out.") from exc

        content = result["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("The model returned empty or invalid content.")
        return content.strip()
