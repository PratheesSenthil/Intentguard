import sys
import json
import os
import csv
import argparse
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from intentguard.validator import validate

def main():
    parser = argparse.ArgumentParser(description="Evaluate IntentGuard.")
    parser.add_argument("label", help="Experiment label")
    parser.add_argument("dataset_path", help="Path to the JSON dataset")
    parser.add_argument("--grounding", choices=["on", "off"], default="on", help="Enable or disable value grounding")
    parser.add_argument("--details", action="store_true", help="Print detailed breakdown of mismatched calls")
    args = parser.parse_args()

    label = args.label
    dataset_path = args.dataset_path
    set_name = os.path.basename(dataset_path)
    grounding = (args.grounding == "on")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)['cases']
        
    total = 0
    matched = 0
    sa = 0
    sb = 0
    
    allow_block_total = 0
    allow_block_matched = 0
    
    cat_total = defaultdict(int)
    cat_matched = defaultdict(int)
    
    current_verdicts = []
    mismatched_case_ids = set()
    detailed_mismatches = []
    
    for case in cases:
        trusted_values = case.get('trusted_values', [])
        try:
            res = validate(
                case['user_message'], 
                case['api_calls'], 
                threshold=0.55, 
                low_conf_floor=0.25, 
                domain='ecommerce',
                trusted_values=trusted_values,
                grounding=grounding
            )
            actual = [v['verdict'] for v in res['verdicts']]
        except Exception as e:
            actual = ['low_confidence'] * len(case['expected_verdicts'])
            
        current_verdicts.extend(actual)
        
        has_mismatch = False
        for i, (exp, act) in enumerate(zip(case['expected_verdicts'], actual)):
            total += 1
            if exp == act:
                matched += 1
                cat_matched[exp] += 1
            else:
                has_mismatch = True
            cat_total[exp] += 1
                
            if exp in ('orphaned', 'injected') and act == 'approved':
                sa += 1
            if exp == 'approved' and act != 'approved':
                sb += 1
                
            allow_block_total += 1
            if (exp == 'approved') == (act == 'approved'):
                allow_block_matched += 1
                
        if has_mismatch:
            mismatched_case_ids.add(case['id'])
            for i, (exp, act) in enumerate(zip(case['expected_verdicts'], actual)):
                if exp != act:
                    detailed_mismatches.append({
                        "case": case,
                        "i": i,
                        "exp": exp,
                        "act": act,
                        "res": res if 'res' in locals() else None
                    })
                
    acc = (matched / total * 100) if total > 0 else 0
    ab_acc = (allow_block_matched / allow_block_total * 100) if allow_block_total > 0 else 0
    
    cache_file = f"results/verdicts_{set_name}.json"
    delta = 0
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            prev_verdicts = json.load(f)
        for c, p in zip(current_verdicts, prev_verdicts):
            if c != p:
                delta += 1
                
    os.makedirs("results", exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(current_verdicts, f)
        
    csv_file = "results/ablation.csv"
    write_header = not os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Label', 'Set', 'Accuracy', 'Safety A', 'Safety B', 'Delta'])
        writer.writerow([label, set_name, f"{acc:.1f}%", sa, sb, delta])
        
    print(f"[{label}] {set_name}:")
    print(f"  Exact Acc: {matched}/{total} ({acc:.1f}%)")
    print(f"  Allow/Block Acc: {allow_block_matched}/{allow_block_total} ({ab_acc:.1f}%)")
    print(f"  Category Acc: " + ", ".join(f"{k}={cat_matched[k]}/{cat_total[k]} ({cat_matched[k]/cat_total[k]*100:.1f}%)" for k in sorted(cat_total.keys())))
    print(f"  Safety A={sa}, Safety B={sb}")
    print(f"  Mismatched Cases: " + ", ".join(sorted(list(mismatched_case_ids))))
    
    if args.details:
        print("\n=== DETAILED MISMATCHES ===")
        for dm in detailed_mismatches:
            c = dm["case"]
            i = dm["i"]
            act = dm["act"]
            exp = dm["exp"]
            res = dm["res"]
            
            call = c["api_calls"][i]
            user_msg = c["user_message"]
            
            print(f"\nCase ID: {c['id']} (Call #{i+1})")
            print(f"User Message: {user_msg}")
            
            params_str = json.dumps(call.get('params', {}))
            print(f"Call: Method={call.get('method')}, Endpoint={call.get('endpoint')}, Params={params_str}, Provenance={call.get('provenance')}")
            
            if res and 'verdicts' in res and i < len(res['verdicts']):
                v_obj = res['verdicts'][i]
                expl = v_obj.get("explanation", {})
                
                print(f"Scores: Cosine={expl.get('cosine_score')}, Verb={expl.get('verb_score')}, Entity={expl.get('entity_score')} -> Total={expl.get('weighted_total')}")
                print(f"Verb Groups: Intent='{expl.get('intent_verb_group')}', Call='{expl.get('call_verb_group')}'")
                print(f"Verb Relation: {expl.get('verb_relation')}")
                print(f"Reason: {expl.get('reason')}")
                print(f"Grounding Object: {json.dumps(expl.get('grounding')) if 'grounding' in expl else 'null'}")
            
            print(f"Expected Verdict: {exp}")
            print(f"Actual Verdict: {act}")


if __name__ == "__main__":
    main()
