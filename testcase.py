#!/usr/bin/env python3
"""
Test your agent against expected answers
Run this from C:\Users\sgarg65
"""

import json
import os
import random
from pathlib import Path

def load_data():
    """Load both questions and expected answers with proper UTF-8 encoding"""
    test_file = Path("cse_476_final_project_test_data.json")
    answers_file = Path("cse_476_final_project_answers.json")
    
    # Check if files exist
    if not test_file.exists():
        print(f"ERROR: {test_file} not found!")
        print(f"Current directory: {os.getcwd()}")
        print("Files in current directory:")
        for f in Path(".").glob("*.json"):
            print(f"  - {f.name}")
        return None, None
    
    if not answers_file.exists():
        print(f"ERROR: {answers_file} not found!")
        return None, None
    
    # Load with utf-8 encoding (fixes the UnicodeDecodeError)
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        print(f"✅ Loaded {len(questions)} questions using UTF-8")
    except UnicodeDecodeError:
        # Try utf-8-sig (handles BOM)
        with open(test_file, 'r', encoding='utf-8-sig') as f:
            questions = json.load(f)
        print(f"✅ Loaded {len(questions)} questions using UTF-8-SIG")
    
    try:
        with open(answers_file, 'r', encoding='utf-8') as f:
            answers = json.load(f)
        print(f"✅ Loaded {len(answers)} answers using UTF-8")
    except UnicodeDecodeError:
        with open(answers_file, 'r', encoding='utf-8-sig') as f:
            answers = json.load(f)
        print(f"✅ Loaded {len(answers)} answers using UTF-8-SIG")
    
    return questions, answers

def normalize(s):
    """Normalize string for comparison"""
    if not s:
        return ""
    import re
    s = str(s).strip().lower()
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def test_sample(sample_size=10):
    """Test on a sample of questions"""
    questions, expected_answers = load_data()
    
    if questions is None:
        return
    
    # Create pairs
    pairs = list(zip(questions, expected_answers))
    
    # Take random sample
    if sample_size > len(pairs):
        sample_size = len(pairs)
    
    sample = random.sample(pairs, sample_size)
    
    print(f"\n{'='*60}")
    print(f"Testing on {len(sample)} random questions")
    print(f"{'='*60}")
    
    for i, (q, expected) in enumerate(sample):
        question_text = q.get("input") or q.get("prompt", "")
        expected_text = expected.get("output", "")
        
        print(f"\n[{i+1}/{len(sample)}]")
        print(f"Question: {question_text[:100]}...")
        print(f"Expected: {expected_text[:100]}...")
        print("-" * 40)
    
    print(f"\n{'='*60}")
    print("To test your actual agent, import your agent function and call it here")
    print(f"{'='*60}")

def test_with_agent(sample_size=5):
    """Test your actual agent on a sample"""
    # Import your agent (adjust import based on your file name)
    try:
        # Try different possible file names
        from cse_476_generate_answers import agent, reset_call_counter, get_call_count
        print("✅ Loaded agent from cse_476_generate_answers.py")
    except ImportError:
        try:
            from generate_answer_template import agent
            print("✅ Loaded agent from generate_answer_template.py")
        except ImportError:
            print("❌ Could not find your agent file!")
            print("Make sure your agent code is in:")
            print("  - cse_476_generate_answers.py OR")
            print("  - generate_answer_template.py")
            return
    
    questions, expected_answers = load_data()
    
    if questions is None:
        return
    
    pairs = list(zip(questions, expected_answers))
    sample = random.sample(pairs, min(sample_size, len(pairs)))
    
    print(f"\n{'='*60}")
    print(f"Testing AGENT on {len(sample)} questions")
    print(f"{'='*60}")
    
    correct = 0
    total_calls = 0
    
    for i, (q, expected) in enumerate(sample):
        question_text = q.get("input") or q.get("prompt", "")
        expected_text = expected.get("output", "")
        
        print(f"\n[{i+1}/{len(sample)}]")
        print(f"Q: {question_text[:80]}...")
        
        try:
            reset_call_counter()
            predicted = agent(question_text)
            calls = get_call_count()
            total_calls += calls
            
            if normalize(predicted) == normalize(expected_text):
                correct += 1
                print(f"✅ CORRECT ({calls} calls)")
            else:
                print(f"❌ INCORRECT ({calls} calls)")
                print(f"   Expected: {expected_text[:80]}...")
                print(f"   Got: {predicted[:80]}...")
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        print("-" * 40)
    
    # Summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {correct}/{len(sample)} ({correct/len(sample)*100:.1f}%)")
    print(f"Avg calls/question: {total_calls/len(sample):.1f}")
    
    if total_calls/len(sample) <= 20:
        print("✅ Within 20 call limit")
    else:
        print("❌ EXCEEDED 20 call limit!")
    
    print(f"{'='*60}")

def check_files():
    """Check what files are in the directory"""
    print(f"\nFiles in {os.getcwd()}:")
    for f in Path(".").glob("*.py"):
        print(f"  📄 {f.name}")
    for f in Path(".").glob("*.json"):
        print(f"  📋 {f.name}")

if __name__ == "__main__":
    print("=" * 60)
    print("CSE 476 - Agent Test Script")
    print("=" * 60)
    print(f"Current directory: {os.getcwd()}")
    
    # Check what files we have
    check_files()
    
    # First, test loading the data
    print("\n" + "=" * 60)
    print("STEP 1: Testing data loading")
    print("=" * 60)
    questions, answers = load_data()
    
    if questions:
        print(f"\n✅ Successfully loaded {len(questions)} questions")
        print(f"✅ Successfully loaded {len(answers)} answers")
        
        # Show first question as example
        print("\n" + "=" * 60)
        print("EXAMPLE FIRST QUESTION")
        print("=" * 60)
        print(f"Question: {questions[0].get('input', 'N/A')[:200]}")
        print(f"Answer: {answers[0].get('output', 'N/A')[:200]}")
        
        # Ask what to do
        print("\n" + "=" * 60)
        print("OPTIONS")
        print("=" * 60)
        print("1. Test data loading only (no agent)")
        print("2. Test your agent on 5 questions")
        print("3. Test your agent on 20 questions")
        
        choice = input("\nEnter choice (1, 2, or 3): ").strip()
        
        if choice == "1":
            test_sample(sample_size=5)
        elif choice == "2":
            test_with_agent(sample_size=5)
        elif choice == "3":
            test_with_agent(sample_size=20)
        else:
            print("Invalid choice")
    else:
        print("\n❌ Could not load data. Check file names and encoding.")