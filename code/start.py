#!/usr/bin/env python3
"""SymptomSense startup script"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    app_dir = Path(__file__).parent
    
    print("🫀 SymptomSense — AI Symptom Tracker")
    print("=" * 40)
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set — AI features disabled")
        print("   Set it with: export ANTHROPIC_API_KEY=your_key_here")
        print()
    else:
        print("✅ Anthropic API key detected")
    
    print("🚀 Starting server on http://localhost:8000")
    print("   Open this URL on your phone or browser")
    print("   Press Ctrl+C to stop")
    print()
    
    os.chdir(app_dir)
    subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])

if __name__ == "__main__":
    main()
