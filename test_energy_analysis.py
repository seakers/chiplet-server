#!/usr/bin/env python3
"""
Test script for enhanced energy analysis in the chat system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.ChatBot.model import ChatBotModel

def test_energy_analysis():
    """Test the enhanced energy analysis functionality."""
    
    # Initialize the chat bot
    chat_bot = ChatBotModel()
    
    # Load a sample point context
    context_file_path = "api/Evaluator/cascade/chiplet_model/dse/results/pointContext/5gpu2attn4sparse1conv.json"
    
    if not os.path.exists(context_file_path):
        print(f"Error: Context file not found at {context_file_path}")
        return
    
    # Add the context
    chat_bot.add_information(context_file_path)
    
    # Test energy-related questions
    test_questions = [
        "What is the bottleneck for energy?",
        "Which chiplet consumes the most energy?",
        "What are the energy bottlenecks in this design?",
        "Show me the highest energy consuming chiplets",
        "What's causing high energy consumption?",
        "Analyze energy distribution across chiplets",
        "Which chiplets are energy inefficient?",
        "What's the energy profile of this design?"
    ]
    
    print("Testing Enhanced Energy Analysis\n")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        print("-" * 30)
        
        try:
            response = chat_bot.get_response(question)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    test_energy_analysis() 