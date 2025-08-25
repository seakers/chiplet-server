#!/usr/bin/env python3
"""
Test script for concise energy bottleneck analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.ChatBot.model import ChatBotModel

def test_concise_energy_analysis():
    """Test the concise energy analysis functionality."""
    
    # Initialize the chat bot
    chat_bot = ChatBotModel()
    
    # Load a sample point context
    context_file_path = "api/Evaluator/cascade/chiplet_model/dse/results/pointContext/5gpu2attn4sparse1conv.json"
    
    if not os.path.exists(context_file_path):
        print(f"Error: Context file not found at {context_file_path}")
        return
    
    # Add the context
    chat_bot.add_information(context_file_path)
    
    # Test the main energy bottleneck question
    question = "What is the bottleneck for energy?"
    
    print("Testing Concise Energy Bottleneck Analysis\n")
    print("=" * 60)
    print(f"Question: {question}")
    print("-" * 60)
    
    try:
        response = chat_bot.get_response(question)
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_concise_energy_analysis() 