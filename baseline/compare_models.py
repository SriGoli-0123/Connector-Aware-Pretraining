#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def generate_comparison(logic_path, baseline_path):
    logic_data = load_json(logic_path)
    base_data = load_json(baseline_path)
    
    logic_results = logic_data['results']
    base_results = base_data['results']
    
    # Header
    comparison = []
    comparison.append("="*80)
    comparison.append("LOGIQA SIDE-BY-SIDE COMPARISON: BASELINE VS. LOGIC CONNECTOR")
    comparison.append("="*80)
    comparison.append(f"Baseline Accuracy: {base_data['accuracy']:.4f} ({base_data['correct_predictions']}/{base_data['total_examples']})")
    comparison.append(f"Logic Connector Accuracy: {logic_data['accuracy']:.4f} ({logic_data['correct_predictions']}/{logic_data['total_examples']})")
    
    delta = logic_data['accuracy'] - base_data['accuracy']
    comparison.append(f"Improvement (Delta): {delta*100:+.2f}%")
    comparison.append("="*80 + "\n")
    
    # Find interesting examples
    improved_examples = []
    degraded_examples = []
    
    for i in range(min(len(logic_results), len(base_results))):
        logic_r = logic_results[i]
        base_r = base_results[i]
        
        # Case: Baseline failed, Connector succeeded
        if not base_r['is_correct'] and logic_r['is_correct']:
            improved_examples.append(logic_r)
            
        # Case: Baseline succeeded, Connector failed
        elif base_r['is_correct'] and not logic_r['is_correct']:
            degraded_examples.append(logic_r)
            
    # Print SUCCESS STORIES (Improved)
    comparison.append("SUCCESS STORIES: Where Logic Connector out-thought the Baseline")
    comparison.append("-" * 60)
    
    for i, ex in enumerate(improved_examples[:10]):  # Show top 10
        comparison.append(f"Example {i+1} (ID: {ex['example_id']})")
        comparison.append(f"Context: {ex['context'][:200]}...")
        comparison.append(f"Question: {ex['question']}")
        comparison.append(f"Correct Answer: {ex['correct_answer']}")
        comparison.append(f"Baseline Prediction: {base_results[ex['example_id']]['predicted_answer']} (WRONG)")
        comparison.append(f"Logic Connector Prediction: {ex['predicted_answer']} (CORRECT)")
        comparison.append(f"Logic Connector Response: {ex['response']}")
        comparison.append("-" * 40)
        
    comparison.append("\n" + "="*80)
    
    # Save to file
    with open("side_by_side_comparison.txt", "w") as f:
        f.write("\n".join(comparison))
    
    print("\n".join(comparison[:10])) # Print summary to terminal
    print(f"\n[INFO] Detailed comparison saved to: side_by_side_comparison.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logic_connector", type=str, required=True)
    parser.add_argument("--baseline", type=str, required=True)
    args = parser.parse_args()
    
    generate_comparison(args.logic_connector, args.baseline)
