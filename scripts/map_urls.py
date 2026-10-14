import json
import re
from pathlib import Path

def main():
    backend_apps_dir = Path("p:/Final year project/ecommerce-enterprise/backend/apps")
    
    action_words = set()
    for url_file in backend_apps_dir.rglob("urls.py"):
        text = url_file.read_text(encoding="utf-8")
        
        # Look for path('...', ...) and url_path='...'
        paths = re.findall(r"path\(['\"]([^'\"]+)['\"]", text)
        url_paths = re.findall(r"url_path=['\"]([^'\"]+)['\"]", text)
        
        for p in paths + url_paths:
            # Drop trailing slash
            if p.endswith("/"):
                p = p[:-1]
            if not p:
                continue
            
            # Get last segment
            segments = p.split("/")
            last_segment = segments[-1]
            
            # Remove <...> like <int:pk>
            if "<" in last_segment:
                continue
                
            # lowercase, hyphens to spaces, singular
            last_segment = last_segment.lower().replace("-", " ")
            if last_segment.endswith("s") and last_segment not in ("address", "status"):
                last_segment = last_segment[:-1]
                
            action_words.add(last_segment)
            
    # Load baseline json
    domain_path = Path("p:/Final year project/intentguard/intentguard/domains/ecommerce.json")
    with open(domain_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    verb_groups = config["verb_groups"]
    entity_vocab = set(config["entity_vocab"])
    
    # 1a & 1b manual changes first (since B builds on them)
    if "close" in verb_groups.get("delete", []):
        verb_groups["delete"].remove("close")
    verb_groups["close"] = ["close", "resolve", "complete", "finish"]
    
    if "disable" in verb_groups.get("delete", []):
        verb_groups["delete"].remove("disable")
    if "enable" in verb_groups.get("apply", []):
        verb_groups["apply"].remove("enable")
    for v in ["show", "view", "display", "read"]:
        if v in verb_groups["get"]:
            verb_groups["get"].remove(v)
            
    verb_groups["restrict"] = ["freeze", "block", "blacklist", "suspend", "lock", "ban", "disable", "flag"]
    verb_groups["restore"] = ["unfreeze", "unblock", "unlock", "reactivate", "enable", "restore", "reinstate"]
    verb_groups["view"] = ["summarize", "summarise", "read", "show", "view", "display", "pull", "review"]
    verb_groups["create"].extend(["reply", "respond", "comment", "note"])
    verb_groups["update"].extend(["mark", "reschedule", "modify items", "reassign"])
    
    # Flatten existing verbs to see what's mapped
    existing_verbs = set()
    for g, verbs in verb_groups.items():
        existing_verbs.update(verbs)
        
    print("NEWLY MAPPED WORDS FROM APP URLS:")
    unmapped = []
    
    for word in sorted(action_words):
        if word in existing_verbs:
            continue
        if word in entity_vocab:
            continue # resource noun
            
        unmapped.append(word)
        
    # Categorise unmapped
    mapping = {
        "advance": "update",
        "approve": "approve", # new
        "assign": "apply",
        "bulk price update": "update",
        "campaign": "campaign", # probably noun
        "chargeback": "refund",
        "damage": "update",
        "deactivate": "delete",
        "discontinue": "delete",
        "escalate": "escalate", # new
        "flag vip": "update",
        "login": "login",
        "me": "get", # nounish
        "merge": "merge", # new
        "method": "update", # update method
        "notify": "send",
        "pack": "pack", # new
        "partial refund": "refund",
        "ready to pack": "get",
        "receive": "receive", # new
        "recent": "get",
        "reopen": "reopen", # new
        "report chargeback": "refund",
        "reset password": "update",
        "resolve": "close",
        "restock alert": "create", # noun
        "risk score": "get",
        "split": "split", # new
        "stack": "apply",
        "staff": "get",
        "store credit": "refund",
        "verify identity": "login",
        "void": "delete",
        "refund": "refund", # move from pay
    }
    
    verb_groups["approve"] = ["approve"]
    verb_groups["refund"] = ["refund", "reimburse", "chargeback", "partial refund", "store credit"]
    if "refund" in verb_groups["pay"]: verb_groups["pay"].remove("refund")
    if "reimburse" in verb_groups["pay"]: verb_groups["pay"].remove("reimburse")
    verb_groups["pack"] = ["pack"]
    verb_groups["merge"] = ["merge"]
    verb_groups["split"] = ["split"]
    verb_groups["escalate"] = ["escalate"]
    verb_groups["reopen"] = ["reopen"]
    verb_groups["receive"] = ["receive"]
    
    for word in unmapped:
        group = mapping.get(word, None)
        if group and group in verb_groups and word not in verb_groups[group]:
            verb_groups[group].append(word)
        print(f"{word} -> {group}")
            
    with open(domain_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()
