#!/usr/bin/env python3
"""
LogiQA dataset loader using existing utils
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.dataset_loader import extract_paper_text, should_keep_paper
from datasets import load_dataset
import json

def create_logiqa_sample():
    """Create sample LogiQA data for testing"""
    return [
        {
            "context": "All birds can fly. Penguins are birds.",
            "question": "Can penguins fly according to the given statements?",
            "options": ["Yes, because penguins are birds", "No, this is a contradiction", "Maybe, it depends", "The statements are unclear"],
            "answer": "A",
            "label": 0
        },
        {
            "context": "If it rains, the ground gets wet. The ground is not wet.",
            "question": "What can we conclude?",
            "options": ["It is raining", "It is not raining", "The ground might be wet", "We cannot conclude anything"],
            "answer": "B", 
            "label": 1
        },
        {
            "context": "All students who study hard pass the exam. John did not pass the exam.",
            "question": "What can we conclude about John?",
            "options": ["John studied hard", "John did not study hard", "John might have studied hard", "We need more information"],
            "answer": "B",
            "label": 1
        },
        {
            "context": "Either the meeting is today or tomorrow. The meeting is not today.",
            "question": "When is the meeting?",
            "options": ["Today", "Tomorrow", "Next week", "We cannot determine"],
            "answer": "B",
            "label": 1
        },
        {
            "context": "If someone is a doctor, then they have a medical degree. Sarah has a medical degree.",
            "question": "What can we conclude about Sarah?",
            "options": ["Sarah is definitely a doctor", "Sarah might be a doctor", "Sarah is not a doctor", "We cannot conclude anything about Sarah being a doctor"],
            "answer": "D",
            "label": 3
        }
    ]

def load_logiqa_dataset():
    """Load LogiQA dataset using a robust fallback strategy"""
    try:
        # Try a few different known slugs and splits
        dataset = None
        for slug in ["heka-ai/logiqa", "tasksource/logiqa", "lucasmccabe/logiqa"]:
            try:
                # Try test split first, then train
                for split_name in ["test", "train"]:
                    try:
                        dataset = load_dataset(slug, split=split_name)
                        print(f"Loaded {len(dataset)} examples from {slug} ({split_name})")
                        break
                    except: continue
                if dataset: break
            except: continue
            
        if not dataset:
            raise ValueError("All HF sources failed")

        def map_columns(example):
            # Handle options being a list or a string (CSV format)
            options = example.get("options", [])
            if isinstance(options, str):
                import ast
                try: options = ast.literal_eval(options)
                except: options = options.split(",")
            
            # Get raw answer/label
            raw_ans = example.get("answer", example.get("label", ""))
            
            # CRITICAL FIX: Convert numerical labels (0, 1, 2, 3) to letters (A, B, C, D)
            if isinstance(raw_ans, (int, float)):
                final_answer = chr(65 + int(raw_ans))
            elif isinstance(raw_ans, str) and raw_ans.isdigit():
                final_answer = chr(65 + int(raw_ans))
            else:
                final_answer = str(raw_ans).upper().strip()
                
            return {
                "context": example.get("context", ""),
                "question": example.get("query", example.get("question", "")),
                "options": options,
                "answer": final_answer
            }
        
        mapped_dataset = dataset.map(map_columns)
        
        # If we had to use 'train', let's just take a representative subset (651 is the standard test size)
        if len(mapped_dataset) > 1000:
            mapped_dataset = mapped_dataset.select(range(651))
            
        print(f"Final evaluation set: {len(mapped_dataset)} examples")
        return mapped_dataset
    except Exception as e:
        print(f"Failed to load from HF: {e}")
        sample_data = create_logiqa_sample()
        print(f"Using sample data: {len(sample_data)} examples")
        return sample_data

if __name__ == "__main__":
    dataset = load_logiqa_dataset()
    print(f"Dataset type: {type(dataset)}")
    if hasattr(dataset, '__len__'):
        print(f"Length: {len(dataset)}")
    if isinstance(dataset, list) and len(dataset) > 0:
        print(f"Sample: {dataset[0]}")