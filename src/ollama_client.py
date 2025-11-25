from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import time
import requests

# Reusing ChatCompletion dataclass
@dataclass
class ChatCompletion:
    content: str
    raw: Dict[str, Any]

class OllamaChatClient:
    """
    Minimal OpenAI-compatible chat client for a local Ollama server.
    
    Ollama specifics:
      - Base URL: http://localhost:11434 (default)
      - Chat path: /api/chat
      - Auth: None (local)
    """

    def __init__(
        self,
        model: str, # Local model is required for local Ollama
        # base_url: str = "http://localhost:11434",
        base_url: str = "http://172.23.224.1:11434", # wsl
        *,
        chat_path: str = "/api/chat",
        timeout: int = 800,
        session: Optional[requests.Session] = None,
        api_key: Optional[str] = None, 
        models_path: str = "/api/tags",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model 
        self.chat_url = f"{self.base_url}{chat_path}"
        self.models_url = f"{self.base_url}{models_path}"
        self.timeout = timeout
        self.session = session or requests.Session()
        
        # Check model exists on machine
        try:
            self.list_models()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Could not connect to Ollama at {self.base_url}. Is it running?")
        except Exception as e:
            print(f"Warning: Could not confirm model {self.model} is installed on Ollama. ({e})")
            
    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def list_models(self) -> Dict[str, Any]:
        """Ollama uses GET /api/tags to list local models."""
        resp = self.session.get(self.models_url, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        response_format_json: bool = False,
        extra: Optional[Dict[str, Any]] = None,
        retry: int = 2,
        retry_backoff_s: float = 1.0,
    ) -> ChatCompletion:
        """
        Send a chat completion request to Ollama's /api/chat endpoint.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        
        # Use JSON flag to force JSON output on smaller models
        if response_format_json:
            payload["format"] = "json"
        
        if top_p is not None:
             payload["options"]["top_p"] = top_p
             
        if extra:
            payload.update(extra)

        last_err = None
        for attempt in range(retry + 1):
            try:
                resp = self.session.post(
                    self.chat_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )

                resp.raise_for_status()
                data = resp.json()
                
                # Ollama response structure
                content = (
                    data.get("message", {})
                    .get("content", "")
                )
                return ChatCompletion(content=content, raw=data)
            except Exception as e:
                last_err = e
                if attempt < retry:
                    time.sleep(retry_backoff_s * (attempt + 1))
                else:
                    raise

        raise RuntimeError(f"Chat failed: {last_err}")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request JSON-only output and parse it."""
        cc = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            response_format_json=True,
            extra=extra,
        )
        try:
            return json.loads(cc.content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Model did not return valid JSON: {e}\n---\n{cc.content}")