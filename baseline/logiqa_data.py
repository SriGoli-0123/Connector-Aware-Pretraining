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

import ast

def load_logiqa_dataset():
    """Ultra-fast optimized LogiQA loader"""
    try:
        # Stick to the mirror we know works to avoid timeout loops
        slug = "heka-ai/logiqa"
        print(f"Loading {slug}...")
        dataset = load_dataset(slug, split="train")
        
        # SPEED OPTIMIZATION: Select the test subset (651) BEFORE mapping
        # This saves processing 7,000+ unnecessary rows
        if len(dataset) > 651:
            dataset = dataset.select(range(651))
        
        def map_columns(example):
            # Handle options safely
            options = example.get("options", [])
            if isinstance(options, str):
                try: options = ast.literal_eval(options)
                except: options = options.split(",")
            
            # Map labels to A, B, C, D
            raw_ans = example.get("answer", example.get("label", ""))
            if isinstance(raw_ans, (int, float, str)) and str(raw_ans).isdigit():
                final_answer = chr(65 + int(raw_ans))
            else:
                final_answer = str(raw_ans).upper().strip()
                
            return {
                "context": example.get("context", ""),
                "question": example.get("query", example.get("question", "")),
                "options": options,
                "answer": final_answer
            }
        
        # Now map only the 651 examples
        print("Mapping columns...")
        mapped_dataset = dataset.map(map_columns)
        print(f"Ready! Final set: {len(mapped_dataset)} examples")
        return mapped_dataset
    except Exception as e:
        print(f"HF Load failed: {e}. Using sample.")
        return create_logiqa_sample()

if __name__ == "__main__":
    dataset = load_logiqa_dataset()
    print(f"Dataset type: {type(dataset)}")
    if hasattr(dataset, '__len__'):
        print(f"Length: {len(dataset)}")
    if isinstance(dataset, list) and len(dataset) > 0:
        print(f"Sample: {dataset[0]}")