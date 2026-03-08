#!/usr/bin/env python3
"""
Quick start script for Interpretable RL Demo

This script provides easy access to the main functionality:
- Run the main training script
- Launch the Streamlit demo
- Run tests
"""

import sys
import subprocess
from pathlib import Path
import argparse


def run_main_script():
    """Run the main interpretable RL script."""
    print("🚀 Running Interpretable RL Main Script...")
    subprocess.run([sys.executable, "0740.py"])


def run_streamlit_demo():
    """Launch the Streamlit demo."""
    print("🎨 Launching Streamlit Demo...")
    subprocess.run(["streamlit", "run", "demo/app.py"])


def run_tests():
    """Run the test suite."""
    print("🧪 Running Test Suite...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])


def install_dependencies():
    """Install project dependencies."""
    print("📦 Installing Dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Interpretable RL Quick Start")
    parser.add_argument(
        "command",
        choices=["run", "demo", "test", "install"],
        help="Command to execute"
    )
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_main_script()
    elif args.command == "demo":
        run_streamlit_demo()
    elif args.command == "test":
        run_tests()
    elif args.command == "install":
        install_dependencies()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🤖 Interpretable Reinforcement Learning")
        print("=" * 50)
        print("Available commands:")
        print("  python quickstart.py run     - Run main training script")
        print("  python quickstart.py demo    - Launch Streamlit demo")
        print("  python quickstart.py test    - Run test suite")
        print("  python quickstart.py install - Install dependencies")
        print("\nFor more information, see README.md")
    else:
        main()
