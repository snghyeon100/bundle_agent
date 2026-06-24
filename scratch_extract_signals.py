import csv
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

csv_path = r"C:\Users\dmlab_student\Desktop\Lab\bundle_agent\results\pog\results_pog_simple_generate_evaluate_decide_C10_T5_20260624_143230_partial.csv"

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    row = next(reader)
    
evidence_json_str = row['simple_signal_final_evidence_json']
evidence = json.loads(evidence_json_str)

for sig in evidence.get('signals', []):
    print(f"Signal Name: {sig['signal_name']}")
    print(f"Description: {sig['description']}")
    print(f"Sources: {sig['sources']}")
    
    # Just show observation for target D (Predicted) and E (True) to see the difference
    print("Candidate D (Pred):", sig['candidate_observations'].get('D'))
    print("Candidate E (True):", sig['candidate_observations'].get('E'))
    print("-" * 50)
