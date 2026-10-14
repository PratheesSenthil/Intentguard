import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intentguard.validator import _extract_intents, _serialise_call
from intentguard.scorer import _cosine_score, _verb_score, _entity_score, _extract_verb, _tokenise
from intentguard.embedder import encode
from intentguard.domain_config import load_domain_config

def main():
    json_path = Path(__file__).resolve().parent.parent / "tests" / "labeled_ecommerce_set.json"
    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
        
    domain_config = load_domain_config("ecommerce")
    
    target_calls = {
        "approved_02": [1],
        "approved_03": [0],
        "approved_04": [1],
        "approved_08": [0, 1],
        "injected_01": [0],
        "injected_02": [0],
        "injected_03": [1],
        "orphaned_03": [2],
        "orphaned_05": [1]
    }
    
    print("=========================================")
    print("DETAILED MISMATCH BREAKDOWN")
    print("=========================================\n")
    
    for case in cases:
        cid = case["id"]
        if cid not in target_calls:
            continue
            
        intents = _extract_intents(case["user_message"])
        
        for call_idx in target_calls[cid]:
            raw_call = case["api_calls"][call_idx]
            serialised_call = _serialise_call(raw_call)
            
            # Find best matching intent
            best_score = -1.0
            best_intent = ""
            for intent in intents:
                # Calculate sub-scores manually to get the total
                v_intent = encode(intent)
                v_call = encode(serialised_call)
                cos = _cosine_score(v_intent, v_call)
                vrb = _verb_score(intent, serialised_call, domain_config)
                ent = _entity_score(intent, serialised_call, domain_config)
                score = 0.60 * cos + 0.25 * vrb + 0.15 * ent
                if score > best_score:
                    best_score = score
                    best_intent = intent
                    best_cos = cos
                    best_vrb = vrb
                    best_ent = ent
            
            iv = _extract_verb(_tokenise(best_intent), domain_config["_verb_canonical"])
            cv = _extract_verb(_tokenise(serialised_call), domain_config["_verb_canonical"])
            
            print(f"[{cid} | call {call_idx}]")
            print(f"Best Intent Text: {best_intent!r}")
            print(f"Serialised Call:  {serialised_call!r}")
            print(f"Scores: Cosine={best_cos:.4f}, Verb={best_vrb:.4f}, Entity={best_ent:.4f} (Total: {best_score:.4f})")
            print(f"Canonical Verbs: Intent='{iv}', Call='{cv}'")
            print("-" * 50)
            
    print("\n=========================================")
    print("VERB RESOLUTION IN ecommerce.json")
    print("=========================================\n")
    
    test_verbs = ["close", "delete", "summarize", "pull", "look", "add"]
    for v in test_verbs:
        canonical = domain_config["_verb_canonical"].get(v, None)
        print(f"'{v}' resolves to -> {canonical!r}")

if __name__ == "__main__":
    main()
