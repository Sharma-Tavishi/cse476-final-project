#!/usr/bin/env python3
"""
Test the agent on a small sample of questions.
"""

import json
import random
from pathlib import Path
from cse_476_agent import agent, reset_call_counter, get_call_count, normalize

def test_sample(sample_size=20):
    """Test agent on random sample."""
    
    with open("cse_476_final_project_test_data.json", 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    sample = random.sample(questions, min(sample_size, len(questions)))
    
    print(f"\nTesting on {len(sample)} random questions")
    print("=" * 60)
    
    total_calls = 0
    
    for i, q in enumerate(sample):
        question_text = q.get("input") or q.get("prompt", "")
        
        print(f"\n[{i+1}/{len(sample)}]")
        print(f"Q: {question_text[:80]}...")
        
        reset_call_counter()
        answer = agent(question_text)
        calls = get_call_count()
        total_calls += calls
        
        print(f"A: {answer[:80]}...")
        print(f"Calls: {calls}")
        print("-" * 40)
    
    print(f"\n{'='*60}")
    print(f"Average calls/question: {total_calls/len(sample):.1f}")
    print(f"Within 20 limit: {'✅' if total_calls/len(sample) <= 20 else '❌'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_sample(sample_size=20)