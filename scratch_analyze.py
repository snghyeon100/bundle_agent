import csv

csv_path = r"C:\Users\dmlab_student\Desktop\Lab\bundle_agent\results\pog\results_pog_simple_generate_evaluate_decide_C10_T5_20260624_143230_partial.csv"

try:
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Total rows: {len(rows)}")
    
    hits = sum(1 for r in rows if r.get('hit') == 'True' or r.get('hit') == '1' or r.get('hit') == 'True' or str(r.get('hit')).lower() == 'true')
    valid = sum(1 for r in rows if str(r.get('simple_signal_rule_validation_passed')).lower() == 'true')
    
    print(f"Hits: {hits} / {len(rows)}")
    print(f"Valid ratio (rule validation passed): {valid} / {len(rows)}")
    
    if rows:
        last_row = rows[-1]
        print(f"Overall Hit Rate: {last_row.get('overall_hit_rate', 'N/A')}")
        print(f"Overall Valid Ratio: {last_row.get('overall_valid_ratio', 'N/A')}")
        print(f"Valid Only Hit Rate: {last_row.get('valid_only_hit_rate', 'N/A')}")
        
        print("\nSample Predictions:")
        for row in rows:
            print(f"  Bundle {row.get('bundle_id')}: True={row.get('true_option_char')}, Pred={row.get('prediction')}, Hit={row.get('hit')}")

except Exception as e:
    print(f"Error: {e}")
