#!/usr/bin/env python3
"""
Test script for runtime bottleneck analysis in the chat system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.ChatBot.model import ChatBotModel

def test_runtime_analysis():
    """Test the runtime analysis functionality."""
    
    # Initialize the chat bot
    chat_bot = ChatBotModel()
    
    # Load a sample point context
    context_file_path = "api/Evaluator/cascade/chiplet_model/dse/results/pointContext/5gpu2attn4sparse1conv.json"
    
    if not os.path.exists(context_file_path):
        print(f"Error: Context file not found at {context_file_path}")
        return
    
    # Add the context
    chat_bot.add_information(context_file_path)
    
    # Test runtime-related questions
    test_questions = [
        "What is the bottleneck for runtime?",
        "Which chiplet is the slowest?",
        "What are the runtime bottlenecks in this design?",
        "Show me the slowest chiplets",
        "What's causing slow execution?",
        "Analyze execution time distribution across chiplets",
        "Which chiplets are performance bottlenecks?",
        "What's the performance profile of this design?"
    ]
    
    print("Testing Runtime Bottleneck Analysis\n")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\nTest {i}: {question}")
        print("-" * 60)
        
        try:
            response = chat_bot.get_response(question)
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    test_runtime_analysis() 