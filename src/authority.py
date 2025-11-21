# authorities.py
import requests
from typing import List, Dict, Any, Optional

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
LOBID_GND_SEARCH_URL = "https://lobid.org/gnd/search"

USER_AGENT = (
    "LassbergEntityLinker/0.1 "
    "(https://github.com/michaelscho/lassberg; mailto:michael.schonhardt@tu-darmstadt.de)"
)

WIKIDATA_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

WIKIDATA_SPARQL_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/sparql-results+json",
}

def normalize_type(ent_type: str) -> str:
    if not ent_type:
        return "UNKNOWN"
    t = ent_type.strip().upper()
    if t in {"PER", "PERSON"}:
        return "PER"
    if t in {"LOC", "LOCATION", "PLACE"}:
        return "LOC"
    if t in {"WORK", "TITLE"}:
        return "WORK"
    return t

def gnd_type_filter(ent_type: str) -> Optional[str]:
    ent_type = normalize_type(ent_type)
    if ent_type == "PER":
        return "type:Person"
    if ent_type == "LOC":
        return "type:PlaceOrGeographicName"
    if ent_type == "WORK":
        return "type:Work"
    return None

WIKIDATA_ALLOWED_INSTANCE_OF = {
    "PER": {"Q5"},
    "LOC": {
        "Q486972", "Q515", "Q6256", "Q56061", "Q82794",
        "Q2221906", "Q23442", "Q8502", "Q4022",
    },
    "WORK": {"Q17537576", "Q47461344", "Q571", "Q7725634", "Q13442814"},
}

def clean_label(label: str, type: str) -> str:
    if type == "PER":
        if label.endswith("ius"):
            label = label[:-3]
        
        if label.endswith("us"):
            label = label[:-2]
        label = label.replace("v.", "von")
        label = label.replace("H.", "Herr")
        label = label.replace("d.", "der")
        print(label)
        
    return label.strip()

def query_wikidata_raw(label: str, limit: int = 30) -> List[Dict[str, Any]]:
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "de",
        "uselang": "de",
        "format": "json",
        "type": "item",
        "limit": limit,
    }
    r = requests.get(
        WIKIDATA_SEARCH_URL,
        params=params,
        headers=WIKIDATA_HEADERS,
        timeout=10,
    )
    r.raise_for_status()
    data = r.json()
    results = []
    for item in data.get("search", []):
        qid = item.get("id")
        results.append({
            "id": qid,
            "uri": f"https://www.wikidata.org/entity/{qid}",
            "label": item.get("label"),
            "description": item.get("description"),
            "aliases": item.get("aliases", []),
            "search_match": item.get("match", {}),
        })
    return results[:limit]

def fetch_wikidata_instance_of(qids: List[str]) -> Dict[str, List[str]]:
    if not qids:
        return {}
    values_clause = " ".join(f"wd:{qid}" for qid in qids)
    query = f"""
    SELECT ?item ?inst WHERE {{
      VALUES ?item {{ {values_clause} }}
      OPTIONAL {{ ?item wdt:P31 ?inst. }}
    }}
    """
    headers = {"Accept": "application/sparql-results+json"}
    params = {"query": query}
    r = requests.get(
        WIKIDATA_SPARQL_URL,
        headers=WIKIDATA_SPARQL_HEADERS,
        params=params,
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    inst_map: Dict[str, List[str]] = {qid: [] for qid in qids}
    for b in data.get("results", {}).get("bindings", []):
        item_uri = b["item"]["value"]
        qid = item_uri.rsplit("/", 1)[-1]
        inst_uri = b.get("inst", {}).get("value")
        if inst_uri:
            inst_qid = inst_uri.rsplit("/", 1)[-1]
            inst_map.setdefault(qid, []).append(inst_qid)
    return inst_map

def filter_wikidata_by_type(
    candidates: List[Dict[str, Any]],
    ent_type: str
) -> List[Dict[str, Any]]:
    ent_type_norm = normalize_type(ent_type)
    allowed = WIKIDATA_ALLOWED_INSTANCE_OF.get(ent_type_norm)
    if not allowed or not candidates:
        return candidates
    qids = [c["id"] for c in candidates if c.get("id")]
    inst_map = fetch_wikidata_instance_of(qids)
    filtered = []
    for c in candidates:
        qid = c.get("id")
        insts = inst_map.get(qid, [])
        if any(i in allowed for i in insts):
            c = dict(c)
            c["instance_of"] = insts
            filtered.append(c)
    if filtered:
        return filtered
    return candidates

def query_wikidata(label: str, ent_type: str, limit: int = 30) -> List[Dict[str, Any]]:
    raw = query_wikidata_raw(label, limit=limit)
    return filter_wikidata_by_type(raw, ent_type)

def query_gnd(label: str, ent_type: str, limit: int = 30) -> List[Dict[str, Any]]:
    params = {
        "q": label,
        "format": "json",
        "size": limit,
    }
    type_filter = gnd_type_filter(ent_type)
    if type_filter:
        params["filter"] = type_filter
    r = requests.get(LOBID_GND_SEARCH_URL, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    members = data.get("member", [])
    results = []
    for m in members:
        gnd_id = m.get("gndIdentifier")
        results.append({
            "id": gnd_id,
            "uri": m.get("@id"),
            "label": m.get("preferredName"),
            "variantNames": m.get("variantName", []),
            "type": m.get("@type"),
            "raw": m,
        })
    return results[:limit]

def lookup_authorities(label: str, ent_type: str, limit: int = 30) -> Dict[str, Any]:
    ent_type_norm = normalize_type(ent_type)
    label = clean_label(label, ent_type_norm)
    wikidata_cands = query_wikidata(label, ent_type_norm, limit=limit)
    gnd_cands = query_gnd(label, ent_type_norm, limit=limit)
    return {
        "query": label,
        "type": ent_type_norm,
        "wikidata": wikidata_cands,
        "gnd": gnd_cands,
    }
