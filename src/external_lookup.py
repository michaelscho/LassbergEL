from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import uuid
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from saia_client import SaiaChatClient 
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from rapidfuzz import fuzz
from pathlib import Path


# CONFIGURATION
WIKIDATA_API_ENDPOINT = "https://www.wikidata.org/w/api.php"
GND_RECONCILE_ENDPOINT = "https://reconcile.gnd.network"


# Local work register cache and search ---
LOCAL_WORK_CACHE: Dict[str, List[Dict[str, Any]]] = {}

def _load_work_jsonl(path: str, source: str) -> List[Dict[str, Any]]:
    """Loads entities from a JSONL file and formats them for search."""
    entries = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Standardize the output format to match candidate structure
                entries.append({
                    "label": data.get("title", ""),
                    "alt_titles": data.get("alt_titles", ""), # Keep alt titles for better fuzzy search
                    "id": data.get("id", str(uuid.uuid4())), # Use original ID or generate new
                    "source": source,
                    "ref": data.get("url"),
                })
    except Exception as e:
        print(f"   -> WARNING: Could not load local work file {source}: {e}")
    return entries

def _search_local_work_cache(query_text: str, work_files: Dict[str, str]) -> List[Dict[str, Any]]:
    """Searches loaded local work files using fuzzy matching."""
    
    global LOCAL_WORK_CACHE
    if not LOCAL_WORK_CACHE:
        # load all defined files only on first call
        for source, path in work_files.items():
            if Path(path).exists():
                LOCAL_WORK_CACHE[source] = _load_work_jsonl(path, source)
            
    best_candidates = []
    
    # Simple fuzzy search across all loaded sources
    for source, entries in LOCAL_WORK_CACHE.items():
        for entry in entries:
            # Check title similarity
            score = fuzz.ratio(query_text.lower(), entry['label'].lower())
            
            # Check alt_titles similarity
            if entry.get("alt_titles"):
                 score = max(score, fuzz.token_set_ratio(query_text.lower(), entry['alt_titles'].lower()))

            # threshold
            if score >= 75: 
                # Then good candidate
                best_candidates.append({
                    "label": entry['label'],
                    "id": entry['id'],
                    "source": entry['source'],
                    "score": score,
                    "ref": entry['ref'],
                })

    # Sort by score and return the best
    best_candidates.sort(key=lambda x: x['score'], reverse=True)
    return best_candidates[:5]

def _create_retry_session(retries=3, backoff_factor=0.5, status_forcelist=(500, 502, 503, 504)):
    """Creates a requests session with retries and exponential backoff for server errors."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(['GET', 'POST'])
    )

    # Mount the adapter for both HTTP and HTTPS requests
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Wikidata
def _search_wikidata_api(query_text: str, entity_type: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Searches Wikidata entities using the MediaWiki Action API (action=wbsearchentities).
    This is much faster and more reliable for text matching than SPARQL.
    """
    params = {
        "action": "wbsearchentities",
        "search": query_text,
        "language": "de",
        "format": "json",
        "limit": limit,
        "type": "item" 
    }
    
    # Q5=human (person), Q486972=settlement (place/loc), Q47461344=literary work, Q1668602=corporate body
    TYPE_MAP = {
        "PER": "Q5", "LOC": "Q486972", "LIT_WORK": "Q47461344", "ORG": "Q1668602"
    }
    
    results = []

    headers = {
        "User-Agent": "LassbergEL/1.0 (historical entity linking; michael.schonhardt@tu-darmstadt.de)" 
    }
    session = _create_retry_session()

    try:
        response = session.get(WIKIDATA_API_ENDPOINT, params=params, headers=headers, timeout=50)
        response.raise_for_status()
        data = response.json()
        
        # Wikidata Debug
        print(f"   [WD API DEBUG] Raw API response for '{query_text}':")
        print(json.dumps(data.get("search", []), indent=2)[:1000] + "...") 
        # --------------------------

        for entity in data.get('search', []):
            entity_id = entity.get('id')
            
            results.append({
                "label": entity.get('label'),
                "id": entity_id, 
                "source": "Wikidata"
            })
            
    except Exception as e:
        # This will now only be reached after 3 retries have failed.
        print(f"   -> WARNING: Wikidata API query FAILED after retries for '{query_text}': {type(e).__name__} ({e})")
        return []
    finally:
        session.close()

    return results

# GND

def _search_gnd(query_text: str, entity_type: str, limit: int = 5) -> List[Dict[str, Any]]:
    type_map = {
        "person": "DifferentiatedPerson",
        "corporatebody": "CorporateBody",
        "work": "Work",
        "place": "PlaceOrGeographicName",
    }
    gnd_type = type_map.get(entity_type.lower())
    if not gnd_type:
        return []

    reconciliation_queries = {
        "q1": {
            "query": query_text,
            "type": gnd_type,
            "limit": limit,
        }
    }

    params = {"queries": json.dumps(reconciliation_queries)}
    results: List[Dict[str, Any]] = []

    try:
        response = requests.get(GND_RECONCILE_ENDPOINT, params=params, timeout=20)
        print(f"   [GND DEBUG] URL: {response.url}")
        print(f"   [GND DEBUG] Raw GND response for '{query_text}' (Type: {entity_type}):")
        print(response.text[:1000] + "...")
        response.raise_for_status()

        data = response.json()
        batch = data.get("q1", {})
        for result in batch.get("result", []):
            if result.get("match"):
                results.append({
                    "label": result["name"],
                    "id": result["id"],
                    "source": "GND",
                    "score": result.get("score", 0.0),
                })

    except Exception as e:
        print(f"   -> WARNING: GND query failed for '{query_text}': {e}")
        return []

    return results

def _lookup_person_external(text: str) -> List[Dict[str, Any]]:
    """Queries Wikidata and GND for a Person/Organization."""
    print(f"   -> Searching for PER/ORG: '{text}' in Wikidata & GND...")
    
    wikidata_results = _search_wikidata_api(text, entity_type="PER", limit=3)
    gnd_results = _search_gnd(text, "Person", limit=3)
    
    return wikidata_results + gnd_results

def _lookup_place_external(text: str) -> List[Dict[str, Any]]:
    """Queries Wikidata and GND for a Place."""
    print(f"   -> Searching for LOC: '{text}' in Wikidata & GND...")
    
    wikidata_results = _search_wikidata_api(text, entity_type="LOC", limit=3)
    gnd_results = _search_gnd(text, "Place", limit=3)

    return wikidata_results + gnd_results

def _lookup_work_external(text: str, local_work_files: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Queries local JSONL files FIRST, then falls back to GND for Work/Organization.
    """
    print(f"   -> Searching for LIT/BIBL: '{text}' in Local Cache & GND...")
    
    # search local cache
    local_results = _search_local_work_cache(text, local_work_files)
    
    if local_results:
        return local_results 

    # Fallback to GND
    
    # Target 1: Works/Literature/Manuscripts
    work_results = _search_gnd(text, "Work", limit=3)
    
    # Target 2: Organizations (for corporate bodies such as "Buchhandlung")
    org_results = _search_gnd(text, "CorporateBody", limit=2)
    
    return work_results + org_results


# TODO LLM Adjudication

def _build_llm_external_choice_prompt(entity: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Builds the prompt for the LLM to choose among external candidates."""
    SYS_MSG = "You are an expert entity resolver. Given an entity text and a list of candidates from external registers (Wikidata, GND), choose the most appropriate external ID. If none are suitable, respond with 'none'."
    
    candidate_list = "\n".join([
        f"- {c['source']} ID: {c['id']}, Label: {c['label']} (GND: {c.get('gnd_id', 'N/A')})"
        for c in candidates
    ])
    
    # Use normalized_text if available for the prompt presentation
    text_to_present = entity.get("normalized_text", entity.get("text", ""))

    user_msg = (
        f"ENTITY: {text_to_present}\n"
        f"LABEL: {entity.get('target_label')}\n"
        f"EXTERNAL CANDIDATES:\n{candidate_list}\n\n"
        "Respond with the chosen ID (e.g., Q1234 or 118540212) or 'none' in a JSON object: {'pick': 'ID', 'confidence': 0.9}"
    )
    
    return [
        {"role": "system", "content": SYS_MSG},
        {"role": "user", "content": user_msg},
    ]

def _ask_llm_external_pick(llm_client: SaiaChatClient, entity: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    """Sends prompt to LLM and returns selected ID/source."""
    
    msgs = _build_llm_external_choice_prompt(entity, candidates)
    
    try:
        data = llm_client.chat_json(msgs, temperature=0.0, max_tokens=512)
        
        pick_id = str(data.get("pick", "none")).strip()
        conf = float(data.get("confidence", 0.0))
        
        if pick_id and pick_id.lower() != "none":
            selected_candidate = next((c for c in candidates if c['id'] == pick_id), None)
            
            if selected_candidate:
                return {
                    "external_id": pick_id,
                    "external_source": selected_candidate['source'],
                    "confidence": conf,
                    "gnd_id": selected_candidate.get('gnd_id')
                }
        return None
        
    except Exception as e:
        print(f"   -> LLM external pick failed for '{entity.get('normalized_text', entity.get('text'))}': {e}")
        return None


def lookup_unlinked_entities(linked_data: Dict[str, Any], llm_client: SaiaChatClient) -> Dict[str, Any]:
    """
    Identifies unlinked entities, performs external lookup, and adjudicates.
    """
    
    # Combine the main entities list and the entities added by the LLM
    all_entities = linked_data.get("entities", []) + linked_data.get("added_by_llm", [])
    
    # Create a new key for internally registered entities
    if "internal_register" not in linked_data:
        linked_data["internal_register"] = []
    
    print(f"\n--- Starting External Lookup for Letter: {linked_data.get('letter_id', 'N/A')} ---")

    for entity in all_entities:
        # Check if the entity is already linked to the local register
        if entity.get("links", {}).get("register", None) is not None:
            continue

        # Use the normalized text for the lookup
        text_to_lookup = entity.get("normalized_text", entity.get("text", ""))
        raw_label = entity.get("target_label") or entity.get("label") or ""
        
        # Skip TIME, MISC, etc.
        if raw_label in ["TIME", "MISC", "OTHER"]:
            print(f"   -> SKIPPED (TIME/OTHER): '{text_to_lookup}' (Label: {raw_label})")
            continue 
        
        # Route based on LLM-adjudicated type
        
        candidates = [] 

        if raw_label in ["PER", "ORG"]:
            candidates = _lookup_person_external(text_to_lookup)
        elif raw_label == "LOC":
            candidates = _lookup_place_external(text_to_lookup)
        elif raw_label in ["LIT_WORK", "LIT_OBJECT", "BIBL", "MANUSCRIPT"]:
            candidates = _lookup_work_external(text_to_lookup)
        
        
        if not candidates:
            print(f"   -> Lookup Failed: No external candidates found for '{text_to_lookup}'.")
            continue 

        # LLM Adjudication and Internal Registration
        
        external_link = _ask_llm_external_pick(llm_client, entity, candidates)
        
        if external_link:
            # Safety: Generate internal ID for the new entry
            unique_id = f"NEW_{raw_label}_{str(uuid.uuid4())[:8]}"
            
            # Add entry to the internal register log
            linked_data["internal_register"].append({
                "xml_id": unique_id,
                "text": entity.get("text", ""),
                "normalized_text": text_to_lookup,
                "target_label": raw_label,
                **external_link,
            })
            
            # Update the entitys links with the new internal ID
            # external ID as primary link reference
            entity["links"] = {
                "register": {"kind": "external_new", "xml_id": unique_id, "ref": external_link['external_id']},
                "decision": {"method": "llm_external_adjudication", "score": external_link['confidence'], "external_source": external_link['external_source']},
            }
            print(f"   -> SUCCESS: Linked '{text_to_lookup}' to {external_link['external_source']} ID: {external_link['external_id']}")
            
    print("--- External Lookup Complete ---")
    return linked_data