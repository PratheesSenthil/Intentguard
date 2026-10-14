import json
import sys
from pathlib import Path
import copy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intentguard.validator import validate

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    json_path = Path(__file__).resolve().parent.parent / "tests" / "labeled_ecommerce_set.json"
    cases = load_json(json_path)["cases"]
        
    domain_config_path = Path(__file__).resolve().parent.parent / "intentguard" / "domains" / "ecommerce.json"
    current_config_dict = load_json(domain_config_path)
    
    # Pre-compute internal caches for the current config manually to match domain_config.py's behaviour
    verb_canonical_current = {}
    for canonical, synonyms in current_config_dict["verb_groups"].items():
        for syn in synonyms:
            verb_canonical_current[syn] = canonical
    current_config_dict["_verb_canonical"] = verb_canonical_current
    current_config_dict["_entity_set"] = frozenset(current_config_dict["entity_vocab"])
        
    # Run 1 config (Original)
    original_config_dict = copy.deepcopy(current_config_dict)
    if "validate" in original_config_dict["verb_groups"]["get"]:
        original_config_dict["verb_groups"]["get"].remove("validate")
        # rebuild cache
        verb_canonical_original = {}
        for canonical, synonyms in original_config_dict["verb_groups"].items():
            for syn in synonyms:
                verb_canonical_original[syn] = canonical
        original_config_dict["_verb_canonical"] = verb_canonical_original
        
    # --- RUN 1 ---
    print("========================================")
    print("RUN 1: ORIGINAL SETTINGS (0.50 / 0.35, no 'validate' synonym)")
    print("========================================")
    
    run1_results = {}
    matched_1 = 0
    total_1 = 0
    
    for case in cases:
        result = validate(
            user_message=case["user_message"],
            api_calls=case["api_calls"],
            threshold=0.50,
            low_conf_floor=0.35,
            domain=original_config_dict
        )
        actual_verdicts = [v["verdict"] for v in result["verdicts"]]
        actual_scores = [v["score"] for v in result["verdicts"]]
        match_status = "MATCH" if actual_verdicts == case["expected_verdicts"] else "MISMATCH"
        
        run1_results[case["id"]] = {
            "verdicts": actual_verdicts,
            "scores": actual_scores,
            "match_status": match_status
        }
        
        print(f"\nCase: {case['id']} ({case['category']}) -> {match_status}")
        print(f"User Message: {case['user_message']!r}")
        for i, (exp, act, score) in enumerate(zip(case["expected_verdicts"], actual_verdicts, actual_scores)):
            is_match = "[PASS]" if exp == act else "[FAIL]"
            print(f"  Call {i}: Expected={exp:<15} Actual={act:<15} Score={score:.4f}  {is_match}")
                
        for exp, act in zip(case["expected_verdicts"], actual_verdicts):
            total_1 += 1
            if exp == act:
                matched_1 += 1
                
    # --- RUN 2 ---
    print("\n========================================")
    print("RUN 2: CURRENT SETTINGS (0.55 / 0.25, with 'validate' synonym)")
    print("========================================")
    
    run2_results = {}
    matched_2 = 0
    total_2 = 0
    
    for case in cases:
        result = validate(
            user_message=case["user_message"],
            api_calls=case["api_calls"],
            threshold=0.55,
            low_conf_floor=0.25,
            domain=current_config_dict
        )
        actual_verdicts = [v["verdict"] for v in result["verdicts"]]
        actual_scores = [v["score"] for v in result["verdicts"]]
        match_status = "MATCH" if actual_verdicts == case["expected_verdicts"] else "MISMATCH"
        
        run2_results[case["id"]] = {
            "verdicts": actual_verdicts,
            "scores": actual_scores,
            "match_status": match_status
        }
        
        print(f"\nCase: {case['id']} ({case['category']}) -> {match_status}")
        print(f"User Message: {case['user_message']!r}")
        for i, (exp, act, score) in enumerate(zip(case["expected_verdicts"], actual_verdicts, actual_scores)):
            is_match = "[PASS]" if exp == act else "[FAIL]"
            print(f"  Call {i}: Expected={exp:<15} Actual={act:<15} Score={score:.4f}  {is_match}")
                
        for exp, act in zip(case["expected_verdicts"], actual_verdicts):
            total_2 += 1
            if exp == act:
                matched_2 += 1
                
    # --- COMPARISON ---
    print("\n========================================")
    print("SIDE-BY-SIDE COMPARISON OF CHANGES")
    print("========================================")
    print(f"Run 1 Accuracy: {matched_1}/{total_1} ({matched_1/total_1*100:.1f}%)")
    print(f"Run 2 Accuracy: {matched_2}/{total_2} ({matched_2/total_2*100:.1f}%)")
    print("\nCases that changed verdict:")
    
    for case in cases:
        id = case["id"]
        v1 = run1_results[id]["verdicts"]
        v2 = run2_results[id]["verdicts"]
        expected = case["expected_verdicts"]
        
        if v1 != v2:
            print(f"\n- Case: {id} ({case['category']})")
            
            # evaluate if improvement or regression
            m1 = sum(1 for e, a in zip(expected, v1) if e == a)
            m2 = sum(1 for e, a in zip(expected, v2) if e == a)
            
            if m2 > m1:
                status = "IMPROVEMENT"
            elif m1 > m2:
                status = "REGRESSION"
            else:
                status = "CHANGED BUT NET ACCURACY SAME"
                
            print(f"  Status: {status}")
            for i, (exp, a1, a2) in enumerate(zip(expected, v1, v2)):
                if a1 != a2:
                    print(f"  Call {i}: Expected={exp:<15} | Run1={a1:<15} -> Run2={a2:<15}")

if __name__ == "__main__":
    main()
