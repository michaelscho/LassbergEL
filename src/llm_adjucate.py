from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
from pathlib import Path
from saia_client import SaiaChatClient 


script_dir = Path(__file__).parent

file_path_el_adj = script_dir / '..' / 'prompts' / 'system_msg_entity_adjucator.txt'
file_path_el_task = script_dir / '..' / 'prompts' / 'task_entity_adjucate.txt'


with open(file_path_el_adj.resolve(), 'r', encoding='utf-8') as f:
    SYSTEM_PROMPT_BASE = f.read()

with open(file_path_el_task.resolve(), 'r', encoding='utf-8') as f:
    TASK = f.read()

SCHEMA_JSON = {
  "letter_id": "string",
  "title": "string",
  "entities": [
    {
      "text": "string",
      "normalized_text": "string",
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
                            max_per_model: Optional[int],
                            min_score: float = 0.75) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filters NER results by minimum confidence score and then applies the max_per_model cap.
    """
    compact: Dict[str, List[Dict[str, Any]]] = {}
    
    for name, arr in ner_results.items():
        if not isinstance(arr, list):
            compact[name] = []
            continue

        # FILTER: Keep only mentions above the minimum score
        filtered_arr = [mention for mention in arr if mention.get('score', 0.0) >= min_score]

        if max_per_model is not None:
            compact[name] = filtered_arr[:max_per_model]
        else:
            compact[name] = filtered_arr
            
    return compact



def build_messages(letter_id: str,
                   title: str,
                   letter_text: str,
                   ner_results: Dict[str, List[Dict[str, Any]]],
                   *,
                   extra_meta: Optional[Dict[str, Any]] = None,
                   max_per_model: Optional[int] = None,
                   min_ner_score: float = 0.75) -> List[Dict[str, str]]:
    
    compact = _compact_ner_for_prompt(ner_results, max_per_model, min_ner_score)
    
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
    
    user_content = "\n\n".join(user_parts)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    total_prompt_content = SYSTEM_PROMPT + user_content
    total_chars = len(total_prompt_content)
    # Estimate tokens with 4 chars per token
    approx_tokens = int(total_chars / 4)
    
    print("\n--- DEBUG: LLM Prompt Construction ---")
    print(f"Active Filters: Min Score={min_ner_score}, Max Per Model={max_per_model}")
    print(f"Total Prompt Characters: {total_chars}")
    print(f"Approximate Total Tokens (4 char/token): {approx_tokens}")
    print("--- Start System Prompt ---")
    print(SYSTEM_PROMPT)
    print("--- Start User Message (Truncated) ---")
    print(user_content)
    print("--- End Debugging ---\n")


    return messages


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
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Make the LLM call using the client (SAIA or Ollama). 
    Returns parsed JSON from the model, enriched with the model name used for adjudication.
    """
    messages = build_messages(
        letter_id=letter_id,
        title=title,
        letter_text=letter_text,
        ner_results=ner_results,
        extra_meta=extra_meta,
        max_per_model=max_per_model,
    )

    # JSON-only response
    adjudicated_data = client.chat_json(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        extra=extra,
    )
    
    # Get model name from the client instance
    model_used = getattr(client, "model", "unknown-llm-model")

    adjudicated_data["adjudication_model"] = model_used
    
    return adjudicated_data