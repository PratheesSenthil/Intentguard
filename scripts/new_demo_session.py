import os
import time

def new_session():
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    audit_file = os.path.join(logs_dir, "audit.jsonl")
    
    if os.path.exists(audit_file):
        timestamp = int(time.time())
        new_name = os.path.join(logs_dir, f"audit.{timestamp}.jsonl")
        os.rename(audit_file, new_name)
        print(f"Renamed current audit.jsonl to audit.{timestamp}.jsonl")
    
    # Create empty file
    with open(audit_file, "w", encoding="utf-8") as f:
        pass
    print("Started a fresh empty log.")

if __name__ == "__main__":
    new_session()
