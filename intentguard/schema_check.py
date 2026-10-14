import json
import re
import os

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schemas", "ecommerce_openapi.json")

_schema_cache = None

def get_schema():
    global _schema_cache
    if _schema_cache is None:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            _schema_cache = json.load(f)
    return _schema_cache

def _route_to_regex(route: str):
    # Convert openapi path param like {id} to a regex matching a single segment
    pattern = re.sub(r'\{[^}]+\}', r'[^/]+', route)
    return re.compile(f"^{pattern}$")

def check_call(method: str, endpoint: str) -> dict:
    if "{" in endpoint or "}" in endpoint:
        return {"valid": False, "reason": "Path contains unresolved literal placeholders."}
        
    schema = get_schema()
    paths = schema.get("paths", {})
    method_lower = method.lower()
    
    for route, operations in paths.items():
        route_regex = _route_to_regex(route)
        
        if route_regex.match(endpoint):
            if method_lower in operations:
                return {"valid": True, "reason": "Valid route."}
            else:
                return {"valid": False, "reason": f"Method {method.upper()} not allowed."}
                
        # Handle trailing slash missing
        if not endpoint.endswith('/'):
            if route_regex.match(endpoint + '/'):
                if method_lower in operations:
                    return {"valid": True, "reason": "Missing trailing slash normalised."}
                else:
                    return {"valid": False, "reason": f"Method {method.upper()} not allowed."}
                    
    return {"valid": False, "reason": "No such route."}
