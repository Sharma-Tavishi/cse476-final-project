#!/usr/bin/env python3
"""
Run the agent to generate answers for all test questions.
"""

import os
import sys

# Set your API key (or use environment variable)
# Option 1: Set directly (remove before sharing code!)
# os.environ["OPENAI_API_KEY"] = "your-key-here"

# Option 2: Use environment variable (recommended)
if not os.getenv("OPENAI_API_KEY"):
    print("=" * 60)
    print("Please set your API key:")
    print("  Windows PowerShell:")
    print("    $env:OPENAI_API_KEY = 'your-key-here'")
    print("  Mac/Linux:")
    print("    export OPENAI_API_KEY='your-key-here'")
    print("=" * 60)
    sys.exit(1)

# Import and run the agent
from cse_476_agent import main

if __name__ == "__main__":
    main()