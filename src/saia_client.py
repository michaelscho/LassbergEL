from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import os
import json
import time
import requests
import config

@dataclass
class ChatCompletion:
    content: str
    raw: Dict[str, Any]

class SaiaChatClient:
    """
    Minimal OpenAI-compatible chat client for SAIA.
    (Enhanced for server-side debugging.)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        chat_path: str = "/chat/completions",
        models_path: str = "/models",
        timeout: int = 500,
        session: Optional[requests.Session] = None,
        debug_requests: bool = False,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SAIA_BASE_URL") or "https://chat-ai.academiccloud.de/v1").rstrip("/")
        self.api_key = config.saia_api_key
        self.model = model or os.environ.get("SAIA_MODEL") or "llama-3.3-70b-instruct"
        self.chat_url = f"{self.base_url}{chat_path}"
        self.models_url = f"{self.base_url}{models_path}"
        self.timeout = timeout
        self.session = session or requests.Session()
        self.debug_requests = debug_requests

        if not self.api_key:
            raise ValueError("SAIA_API_KEY is not set (env or pass in).")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }


    def list_models(self) -> Dict[str, Any]:
        """SAIA uses POST /v1/models."""
        resp = self.session.post(self.models_url, headers=self._headers(), json={}, timeout=self.timeout)
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
        Send a chat completion request (OpenAI-compatible) with enhanced debugging.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if response_format_json:
            payload["response_format"] = {"type": "json_object"}
        if extra:
            payload.update(extra)

        if self.debug_requests:
            print(f"\n[SAIA DEBUG] Sending request to: {self.chat_url}")
            print(f"[SAIA DEBUG] Model: {self.model}, Tokens: {max_tokens}, Temp: {temperature}")
            truncated_messages = messages[:1] + messages[-1:]
            print(f"[SAIA DEBUG] Messages (Partial): {json.dumps(truncated_messages)[:500]}...")


        last_err = None
        for attempt in range(retry + 1):
            try:
                resp = self.session.post(
                    self.chat_url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                
                if resp.status_code >= 400:
                    err_details = resp.text
                    print(f"\n[SAIA ERROR - ATTEMPT {attempt+1}/{retry+1}] HTTP {resp.status_code} on POST to {self.chat_url}")
                    print(f"[SAIA ERROR] Request failed. Server response body:\n--- START RESPONSE ---\n{err_details[:500]}...\n--- END RESPONSE ---")
                    
                    if resp.status_code == 429 and attempt < retry:
                        time.sleep(retry_backoff_s * (attempt + 1))
                        continue
                        
                    resp.raise_for_status() 
                
                if self.debug_requests:
                    print(f"[SAIA DEBUG] Attempt {attempt+1} successful. Status: {resp.status_code}")


                resp.raise_for_status() 
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                return ChatCompletion(content=content, raw=data)
            except requests.exceptions.HTTPError as e:
                last_err = e
                if attempt < retry:
                    time.sleep(retry_backoff_s * (attempt + 1))
                    continue
                else:
                    raise RuntimeError(f"SAIA Chat failed after {attempt+1} attempts: HTTPError. Check console for server response body.") from e
            except Exception as e:
                last_err = e
                print(f"\n[SAIA ERROR - ATTEMPT {attempt+1}/{retry+1}] Non-HTTP Error: {type(e).__name__} ({e})")
                if attempt < retry:
                    time.sleep(retry_backoff_s * (attempt + 1))
                else:
                    raise RuntimeError(f"SAIA Chat failed after {attempt+1} attempts: Non-HTTP Error.") from e

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
            raise ValueError(f"Model did not return valid JSON: {e}\n--- RAW CONTENT ---\n{cc.content[:1000]}...")