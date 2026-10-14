import json
import sys
from pathlib import Path
import numpy as np

# Add parent directory to sys.path to allow importing intentguard
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intentguard.validator import validate

def main():
    json_path = Path(__file__).resolve().parent.parent / "tests" / "labeled_ecommerce_set.json"
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    cases = data["cases"]
    
    # Grid of thresholds to test
    thresholds = [0.45, 0.50, 0.55, 0.60]
    low_conf_floors = [0.15, 0.20, 0.25, 0.30, 0.35]
    
    best_acc = 0
    best_params = None
    
    print("Running Grid Search for Optimal Thresholds...")
    
    for th in thresholds:
        for lc in low_conf_floors:
            if lc >= th:
                continue
                
            matched = 0
            total = 0
            
            for case in cases:
                expected = case["expected_verdicts"]
                try:
                    result = validate(
                        user_message=case["user_message"],
                        api_calls=case["api_calls"],
                        threshold=th,
                        low_conf_floor=lc,
                        domain="ecommerce"
                    )
                    actual_verdicts = [v["verdict"] for v in result["verdicts"]]
                    for exp, act in zip(expected, actual_verdicts):
                        if exp == act:
                            matched += 1
                        total += 1
                except Exception:
                    pass
            
            acc = matched / total if total > 0 else 0
            print(f"Threshold: {th:.2f}, Low Conf: {lc:.2f} -> Accuracy: {acc*100:.1f}% ({matched}/{total})")
            
            if acc > best_acc:
                best_acc = acc
                best_params = (th, lc)
                
    print("\n===============================")
    print(f"Best Accuracy: {best_acc*100:.1f}%")
    print(f"Optimal Threshold: {best_params[0]:.2f}")
    print(f"Optimal Low Conf Floor: {best_params[1]:.2f}")
    print("===============================")

if __name__ == "__main__":
    main()
