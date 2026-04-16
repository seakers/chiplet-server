"""
ChatBot API endpoints.
Consolidates chatbot endpoints from views.py <source_id data="1" title="views.py" />.
"""
import json
import os

from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.chatbot.bot import ChatBot
from api.config.prompts import FollowUpSuggestions


# Global chatbot instance (maintains conversation history)
chat_bot = None


def get_chatbot(evaluator: str = 'cascade', run_id: str = None) -> ChatBot:
    """Get or create the global chatbot instance."""
    global chat_bot
    if chat_bot is None:
        chat_bot = ChatBot(evaluator=evaluator, run_id=run_id)
    return chat_bot


@api_view(["POST"])
def chat(request):
    """
    Handle chat messages from the user.
    Refactored from views.py <source_id data="1" title="views.py" />.
    """
    try:
        data = json.loads(request.body)
        message = data.get("message")
        evaluator = data.get("evaluator", "cascade")
        run_id = data.get("run_id")
        use_retrieval = data.get("use_retrieval", False)
        
        if not message:
            return Response({"error": "message is required"}, status=400)
        
        bot = get_chatbot(evaluator, run_id)
        
        response = bot.get_response(
            query=message,
            use_retrieval=use_retrieval
        )
        
        # Handle retrieval response format
        if isinstance(response, dict):
            return Response({
                "response": response.get("final_answer", ""),
                "citations": response.get("citations", [])
            })
        
        return Response({"response": response})
        
    except Exception as e:
        print(f"Error in chat: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
def data_mining_followup(request):
    """
    Handle follow-up questions about data mining results with context-aware responses.
    Refactored from views.py <source_id data="1" title="views.py" />.
    """
    try:
        evaluator = request.GET.get("evaluator", "cascade")
        run_id = request.GET.get("run_id")
        
        bot = get_chatbot(evaluator, run_id)
        
        data = json.loads(request.body)
        question = data.get("question")
        data_mining_type = data.get("data_mining_type")  # "rule_mining" or "distance_correlation"
        structured_data = data.get("structured_data", {})
        
        if not question or not data_mining_type:
            return Response({
                "error": "question and data_mining_type are required"
            }, status=400)
        
        # Build context-aware prompt based on data mining type
        if data_mining_type == "rule_mining":
            context = f"Based on the rule mining results: {json.dumps(structured_data)}\n\nUser question: {question}"
        elif data_mining_type == "distance_correlation":
            context = f"Based on the distance correlation analysis: {json.dumps(structured_data)}\n\nUser question: {question}"
        else:
            context = question
        
        response = bot.get_response(context)
        
        return Response({"response": response})
        
    except Exception as e:
        print(f"Error in data_mining_followup: {e}")
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
def add_run_context(request):
    """
    Add optimization run context to the chatbot.
    Refactored from views.py <source_id data="1" title="views.py" />.
    """
    try:
        data = json.loads(request.body)
        summary_text = data.get("summary_text")
        analytics_text = data.get("analytics_text")
        is_comparative = data.get("is_comparative", False)
        
        if not summary_text:
            return Response({"error": "summary_text is required"}, status=400)
        
        bot = get_chatbot()
        
        # Choose appropriate suggestions based on run type
        suggestions = (
            FollowUpSuggestions.COMPARATIVE_RUN 
            if is_comparative 
            else FollowUpSuggestions.SINGLE_RUN
        )
        
        bot.add_run_context(summary_text, analytics_text, suggestions)
        
        return Response({
            "message": "Run context added successfully",
            "suggestions": suggestions
        })
        
    except Exception as e:
        print(f"Error in add_run_context: {e}")
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
def add_point_context(request):
    """
    Add specific design point context to the chatbot.
    Refactored from views.py <source_id data="1" title="views.py" />.
    """
    try:
        data = json.loads(request.body)
        context_file_path = data.get("context_file_path")
        
        if not context_file_path:
            return Response({"error": "context_file_path is required"}, status=400)
        
        if not os.path.exists(context_file_path):
            return Response({"error": f"File not found: {context_file_path}"}, status=404)
        
        bot = get_chatbot()
        bot.add_information(context_file_path)
        
        return Response({"message": "Point context added successfully"})
        
    except Exception as e:
        print(f"Error in add_point_context: {e}")
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
def add_enhanced_insights_context(request):
    """
    Add both summary and detailed context to the AI's conversation history.
    Refactored from views.py <source_id data="1" title="views.py" />.
    """
    try:
        data = json.loads(request.body)
        summary_insights = data.get("summary_insights")
        detailed_context = data.get("detailed_context")
        
        if not summary_insights:
            return Response({"error": "summary_insights is required"}, status=400)
        
        bot = get_chatbot()
        
        # Create enhanced context message
        context_message = f"Here are the insights from the analysis:\n\n{summary_insights}\n\n"
        
        if detailed_context:
            context_message += f"I have detailed analysis available for this design point including:\n"
            context_message += f"- Per-chiplet energy breakdown and execution time\n"
            context_message += f"- Memory access patterns and bottlenecks\n"
            context_message += f"- Work distribution across chiplets\n"
            context_message += f"- Energy efficiency metrics\n\n"
            context_message += f"You can ask detailed questions like:\n"
            context_message += f"- 'What is the energy bottleneck for this design?'\n"
            context_message += f"- 'Which chiplet is consuming the most memory?'\n"
            context_message += f"- 'How is the work distributed across chiplets?'\n"
            context_message += f"- 'What's the energy efficiency of each component?'\n\n"
            context_message += f"Detailed context data is available for analysis."
        else:
            context_message += f"I have this context and can answer follow-up questions about these insights."
        
        # Add the enhanced context to AI memory
        bot.messages.append({
            "role": "assistant",
            "content": context_message
        })
        
        return Response({"message": "Enhanced insights context added successfully"})
        
    except Exception as e:
        print(f"Error in add_enhanced_insights_context: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
def clear_chat_history(request):
    """
    Clear the chatbot conversation history.
    """
    try:
        bot = get_chatbot()
        bot.clear_history()
        return Response({"message": "Chat history cleared"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)