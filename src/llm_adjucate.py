# src/llm_adjudicate_saia.py
"""
LLM adjudication for NER using SAIA client.

- Reuses `SaiaChatClient` (OpenAI-compatible HTTP).
- Preserves offsets by carrying through model spans in `evidence.mentions`.
- Adds a representative `char_span` per consolidated entity when possible.
- Can cap per-model NER items in the prompt via `max_per_model`.

Usage:
    from saia_client import SaiaChatClient
    from llm_adjudicate_saia import adjudicate_entities_with_client

    client = SaiaChatClient()  # uses env: SAIA_API_KEY, SAIA_MODEL, SAIA_BASE_URL
    result = adjudicate_entities_with_client(
        client=client,
        letter_id="1234",
        title="Letter 1234 — Foo to Bar",
        letter_text="...",
        ner_results=ner_bundle,
        max_per_model=500,   # optional, controls prompt size
    )
"""

from __future__ import annotations
import json
from typing import Any, Dict, List, Optional

from saia_client import SaiaChatClient


# Schema and prompt that preserves offsets
with open('..\prompts\system_msg_entity_adjucator.txt', 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT_BASE = f.read()

with open('..\prompts\task_entity_adjucator.txt', 'r', encoding='utf-8') as f:
    TASK = f.read()

SCHEMA_JSON = {
  "letter_id": "string",
  "title": "string",
  "entities": [
    {
      "text": "string",
      "target_label": "PER|LOC|ORG|LIT_WORK|LIT_OBJECT|TIME",
      "added_by_llm": False,
      "char_span": { "start": 0, "end": 0 },
      "evidence": {
        "mentions": [
          {
            "model": "hmbert|flair_base|flair_large|hf_default|llm",
            "text": "string",
            "label": "original model label",
            "score": 0.0,
            "start": 0,
            "end": 0
          }
        ],
        "vote_count": 0
      }
    }
  ],
  "dropped": [
    {
      "text": "string",
      "label": "string",
      "reason": "false positive | out of scope | duplicate shorter span",
      "evidence": {
        "model": "hmbert|flair_base|flair_large|hf_default",
        "score": 0.0
      }
    }
  ]
}

SYSTEM_PROMPT = f"""{SYSTEM_PROMPT_BASE}

{json.dumps(SCHEMA_JSON, ensure_ascii=False, indent=2)}
""".strip()


def _compact_ner_for_prompt(ner_results: Dict[str, List[Dict[str, Any]]],
                            max_per_model: Optional[int]) -> Dict[str, List[Dict[str, Any]]]:
    if not max_per_model:
        return ner_results
    compact: Dict[str, List[Dict[str, Any]]] = {}
    for name, arr in ner_results.items():
        compact[name] = arr[:max_per_model] if isinstance(arr, list) else []
    return compact


def build_messages(letter_id: str,
                   title: str,
                   letter_text: str,
                   ner_results: Dict[str, List[Dict[str, Any]]],
                   *,
                   extra_meta: Optional[Dict[str, Any]] = None,
                   max_per_model: Optional[int] = None) -> List[Dict[str, str]]:
    compact = _compact_ner_for_prompt(ner_results, max_per_model)
    meta = {"letter_id": letter_id}
    if extra_meta:
        meta.update(extra_meta)

    user_parts = [
        "TITLE\n=====\n" + title,
        "METADATA\n========\n" + json.dumps(meta, ensure_ascii=False, indent=2),
        "LETTER\n======\n" + letter_text,
        "NER OUTPUTS (raw)\n=================\n" + json.dumps(compact, ensure_ascii=False),
        TASK 
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def adjudicate_entities_with_client(
    *,
    client: SaiaChatClient,
    letter_id: str,
    title: str,
    letter_text: str,
    ner_results: Dict[str, List[Dict[str, Any]]],
    extra_meta: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    top_p: Optional[float] = None,
    max_per_model: Optional[int] = None,
    # You can pass extra OpenAI/SAIA-compatible flags via `extra`
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Make the SAIA call using your `SaiaChatClient`. Returns parsed JSON from the model.
    """
    messages = build_messages(
        letter_id=letter_id,
        title=title,
        letter_text=letter_text,
        ner_results=ner_results,
        extra_meta=extra_meta,
        max_per_model=max_per_model,
    )

    # Ask SAIA for JSON-only response
    return client.chat_json(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        extra=extra,
    )
