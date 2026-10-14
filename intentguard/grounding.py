import re
from typing import Any

def _is_exempt(v_str: str) -> bool:
    v_lower = v_str.lower()
    if v_lower in ("true", "false"):
        return True
    if " " in v_str:
        return True
    if re.match(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$", v_str):
        return True
    return False

def _extract_values(api_call: dict) -> list[str]:
    vals = []
    
    endpoint = api_call.get("endpoint", "")
    segments = [s for s in endpoint.split("/") if s]
    for seg in segments:
        if any(c.isdigit() for c in seg) and not _is_exempt(seg):
            vals.append(seg)
            
    for bag_key in ("params", "body"):
        bag = api_call.get(bag_key)
        if not bag or not isinstance(bag, dict):
            continue
            
        def _walk(obj: Any):
            if isinstance(obj, dict):
                for v in obj.values():
                    _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)
            else:
                if obj is None: return
                if isinstance(obj, bool): return
                
                v_str = str(obj).strip()
                if not v_str: return
                
                if _is_exempt(v_str):
                    return
                
                is_email = "@" in v_str and "." in v_str
                has_digit = any(c.isdigit() for c in v_str)
                
                if is_email or has_digit:
                    vals.append(v_str)

        _walk(bag)
        
    return vals

def _normalise(val: str) -> str:
    v = val.lower().replace("$", "").replace(",", "")
    try:
        f = float(v)
        f = abs(f)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        pass
    return v

def _get_tokens(text: str) -> set[str]:
    text = text.replace("\u2019", "'")
    text = re.sub(r"(?i)'s\b", " ", text)
    
    tokens = set()
    
    def _strip_punct(s: str) -> str:
        return s.strip(".,;:!?)]}\"'([{")

    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    for m in re.finditer(email_pattern, text):
        tokens.add(_normalise(_strip_punct(m.group(0))))
    text = re.sub(email_pattern, " ", text)
    
    num_pattern = r"-?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\$?\d+(?:\.\d+)?"
    for m in re.finditer(num_pattern, text):
        tokens.add(_normalise(_strip_punct(m.group(0))))
        
    code_pattern = r"[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*"
    for m in re.finditer(code_pattern, text):
        val = _strip_punct(m.group(0))
        if not val: continue
        tokens.add(_normalise(val))
        if '-' in val or '_' in val:
            for sub_m in re.finditer(r"\d+", val):
                tokens.add(_normalise(sub_m.group(0)))
                
    return tokens

def is_grounded(val: str, user_msg_tokens: set[str], trusted_norm: set[str]) -> bool:
    norm_val = _normalise(val)
    return (norm_val in user_msg_tokens) or (norm_val in trusted_norm)

def process_grounding(user_message: str, api_call: dict, trusted_values: list[str]) -> tuple[bool, list[dict]]:
    extracted = _extract_values(api_call)
    
    msg_tokens = _get_tokens(user_message)
    tr_norm = set()
    for t in (trusted_values or []):
        tr_norm.add(_normalise(str(t)))
        
    grounding_objects = []
    all_grounded = True
    
    for val in extracted:
        gr = is_grounded(val, msg_tokens, tr_norm)
        grounding_objects.append({"value": val, "grounded": gr})
        if not gr:
            all_grounded = False
            
    return all_grounded, grounding_objects

