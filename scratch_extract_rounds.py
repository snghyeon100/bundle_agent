import csv
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

csv_path = r"C:\Users\dmlab_student\Desktop\Lab\bundle_agent\results\pog\results_pog_simple_generate_evaluate_decide_C10_T5_20260624_143230_partial.csv"

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    row = next(reader)

# Check the decision prompt
print("==== FINAL DECISION PROMPT ====")
print(row['simple_signal_decision_prompt'])
print("===============================\n")

# Check round traces
round_trace_str = row['simple_signal_round_trace']
round_trace = json.loads(round_trace_str)

for i, rt in enumerate(round_trace):
    print(f"==== ROUND {i+1} ({rt['round_role']}) ====")
    evidence = rt.get('accepted_evidence')
    if evidence and 'signals' in evidence:
        for sig in evidence['signals']:
            print(f"- {sig['signal_name']}")
    else:
        print("No accepted evidence for this round.")
    print()
