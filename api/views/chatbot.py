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


chat_bot = None
_current_evaluator = None
_current_run_id = None

def get_chatbot(evaluator: str = 'cascade', run_id: str = None, force_new: bool = False) -> ChatBot:
    global chat_bot, _current_evaluator, _current_run_id

    evaluator_normalized = evaluator.lower()

    # Only recreate if explicitly forced, or if evaluator changes
    # Do NOT recreate just because run_id changed — that would wipe point context
    evaluator_changed = (_current_evaluator != evaluator_normalized)
    run_changed = (_current_run_id != run_id) and (run_id is not None)

    if force_new or chat_bot is None or evaluator_changed:
        chat_bot = ChatBot(evaluator=evaluator_normalized, run_id=run_id)
        _current_evaluator = evaluator_normalized
        _current_run_id = run_id
        print(f"[get_chatbot] Created new bot (force={force_new}, evaluator_changed={evaluator_changed})")
    elif run_changed:
        # Run changed but same evaluator — update run_id without wiping context
        chat_bot.run_id = run_id
        _current_run_id = run_id
        print(f"[get_chatbot] Updated run_id to {run_id} without reinitializing bot")
    else:
        print(f"[get_chatbot] Reusing existing bot with context intact")

    return chat_bot


@api_view(["POST"])
def chat(request):
    bot = None
    try:
        data = json.loads(request.body) if request.body else {}
        evaluator = data.get("evaluator") or request.GET.get("evaluator", "cascade")
        run_id    = data.get("run_id")    or request.GET.get("run_id")
        objectives= data.get("objectives") or request.GET.get("objectives")
        highlighted_indices = data.get("highlighted_indices") or []
        trace_or_model = data.get("trace_or_model") or request.GET.get("trace_or_model")

        bot = get_chatbot(evaluator, run_id)

        if objectives:
            bot.set_objectives(objectives)

        if trace_or_model:
            bot.set_trace_or_model(trace_or_model)

        bot.highlighted_indices = list(highlighted_indices) if highlighted_indices else []

        role    = data.get("role", "user")
        content = data.get("content", "")
        
        response = bot.get_response(
            query=content,
            use_retrieval=False
        )
        
        # Handle retrieval response format
        if isinstance(response, dict):
            return Response({
                "response": response.get("final_answer", ""),
                "citations": response.get("citations", [])
            })
        
        # Build frontend actions from agent results
        frontend_actions = []
        for agent_name, result in getattr(bot, 'last_agent_results', []):
            if not result.success:
                continue
            if agent_name == 'dcorr_agent':
                frontend_actions.append({
                    'type': 'update_distance_correlation',
                    'data': result.data or {}
                })
            elif agent_name == 'rule_mining_agent':
                frontend_actions.append({
                    'type': 'update_rule_mining',
                    'data': result.data or {}
                })
            elif agent_name == 'highlighting_agent':
                frontend_actions.append({
                    'type': 'highlight_points',
                    'data': result.data or {}
                })
            elif agent_name == 'optimization_agent':
                if result.data and result.data.get('run_id'):
                    frontend_actions.append({
                        'type': 'start_optimization',
                        'data': result.data
                    })
            elif agent_name == 'report_agent':
                frontend_actions.append({
                    'type': 'report_generated',
                    'data': result.data or {}
                })
            elif agent_name == 'comparative_analysis_agent':
                if result.data:
                    frontend_actions.append({
                        'type': 'comparative_analysis_result',
                        'data': result.data
                    })
        bot.last_agent_results = []  # Clear after reading
        print(f"Frontend actions: {frontend_actions}")
        
        return Response({
            "response": response,
            "frontend_actions": frontend_actions
        })
        
    except Exception as e:
        print(f"Error in chat: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
    
    finally:
        if bot:
            bot.last_agent_results = []


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
    try:
        data = json.loads(request.body)
        summary_text = data.get("summary_text")
        analytics_text = data.get("analytics_text")
        is_comparative = data.get("is_comparative", False)
        objectives = data.get("objectives")
        evaluator = data.get("evaluator", "cascade")
        run_id = data.get("run_id", None)
        trace_or_model = data.get("trace_or_model", None)

        if not summary_text:
            return Response({"error": "summary_text is required"}, status=400)

        # Force new bot for a new run — intentional context wipe
        bot = get_chatbot(evaluator=evaluator, run_id=run_id, force_new=True)

        if objectives:
            bot.set_objectives(objectives)

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
    try:
        evaluator = request.data.get("evaluator", "cascade")
        run_id = request.data.get("run_id", None)
        # Force a brand new bot — this is the one place we WANT to wipe context
        bot = get_chatbot(evaluator=evaluator, run_id=run_id, force_new=True)
        return Response({"message": "Chat history cleared"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    

@api_view(["GET"])
def add_info(request):
    model = request.GET.get("model", "CASCADE")
    bot = get_chatbot(evaluator=model.lower())

    if model == "PISTIL":
        num_cus = request.GET.get("num_cus", "0")
        num_tmacs = request.GET.get("num_tmacs", "0")
        mem_buf_cap = request.GET.get("mem_buf_cap", "0")
        net_buf_cap = request.GET.get("net_buf_cap", "0")
        mem_banks_per_group = request.GET.get("mem_banks_per_group", "0")
        mem_ranks = request.GET.get("mem_ranks", "0")
        mem_frac_bank_cap = request.GET.get("mem_frac_bank_cap", "0")
        batch_size = request.GET.get("batch_size", "0")
        kv_cache = request.GET.get("kv_cache", "0")

        # Helper: ensures 1 -> "1.0", 0.25 -> "0.25", 2 -> "2.0"
        def fmt(val):
            n = float(val)
            return f"{n:.1f}" if n == int(n) else str(n)

        chiplet_file_path = [(
            f"api/Evaluator/sim-v2-4-pistil-sim-clean/trace-results/"
            f"llama3-8b-chiplets-{num_cus}-r-{mem_ranks}-bg-{mem_banks_per_group}-"
            f"f-{fmt(mem_frac_bank_cap)}-bs-{batch_size}-kv-{kv_cache}-occ-{float(mem_buf_cap):.2f}.csv"
        ),
        (
            f"api/Evaluator/sim-v2-4-pistil-sim-clean/configs/gen_configs/"
            f"pistil-config-num_cus-{num_cus}-tmacs-{num_tmacs}-mem_buf_cap-{fmt(mem_buf_cap)}-"
            f"net_buf_cap-{fmt(net_buf_cap)}-mem_bank_groups-{mem_banks_per_group}-mem_ranks-{mem_ranks}-"
            f"mem_frac_bank_cap-{fmt(mem_frac_bank_cap)}.json"
        )]
        print(f"Constructed Pistil chiplet file paths: {chiplet_file_path}")
    else:
        gpu    = request.GET.get("gpu", "0")
        attn   = request.GET.get("attn", "0")
        sparse = request.GET.get("sparse", "0")
        conv   = request.GET.get("conv", "0")
        chiplet_file_path = (
            f"api/Evaluator/cascade/chiplet_model/dse/results/"
            f"pointContext/{gpu}gpu{attn}attn{sparse}sparse{conv}conv.json"
        )

    bot.add_information(chiplet_file_path)
    bot.messages.append({
        "role": "assistant",
        "content": "I have received context on this design! I am ready to answer questions about it."
    })
    return Response({"message": "Information added successfully."})


@api_view(["GET"])
def get_chat_response(request):
    """
    GET-based chat endpoint (legacy compatibility).
    Mirrors get_chat_response from views.py [1].
    """
    content   = request.GET.get("content")
    role      = request.GET.get("role", "user")
    evaluator = request.GET.get("evaluator", "cascade")
    run_id    = request.GET.get("run_id", None)
    use_retrieval = request.GET.get("use_retrieval", "false").lower() in ("1", "true", "yes")
    trace_or_model = request.GET.get("trace_or_model", None)
    # Support objectives passed as either a single param or as an array (objectives[])
    # e.g. ?objectives=foo or ?objectives[]=a&objectives[]=b
    if "objectives[]" in request.GET:
        objectives = request.GET.getlist("objectives[]")
    else:
        # getlist will return [] for missing keys, so try both
        objectives = request.GET.getlist("objectives") or request.GET.get("objectives")    

    if not content:
        return Response({"error": "content is required"}, status=400)

    bot = get_chatbot(evaluator, run_id)

    if objectives:
        bot.set_objectives(objectives)

    if trace_or_model:
        bot.set_trace_or_model(trace_or_model)

    try:
        # Collect optional filters
        filters = {}
        for key in ["trace", "doc_type"]:
            val = request.GET.get(key)
            if val:
                filters[key] = val

        # Try retrieval-enabled response first if requested
        if use_retrieval:
            try:
                rag_result = bot.get_response(query=content, role=role, use_retrieval=True, filters=filters or None)
                if isinstance(rag_result, dict) and rag_result.get("final_answer"):
                    return Response({
                        "response": rag_result["final_answer"],
                        "citations": rag_result.get("citations", [])
                    })
            except Exception:
                # fall through to non-retrieval
                pass

        # Non-retrieval / standard conversational response
        response = bot.get_response(query=content, role=role, use_retrieval=False)

        # If bot returned a retrieval-like dict, handle it
        if isinstance(response, dict):
            return Response({
                "response": response.get("final_answer", ""),
                "citations": response.get("citations", [])
            })

        # Build frontend actions from last agent results (if any)
        frontend_actions = []
        for agent_name, result in getattr(bot, 'last_agent_results', []):
            if not result.success:
                continue
            if agent_name == 'dcorr_agent':
                frontend_actions.append({
                    'type': 'update_distance_correlation',
                    'data': result.data or {}
                })
            elif agent_name == 'rule_mining_agent':
                frontend_actions.append({
                    'type': 'update_rule_mining',
                    'data': result.data or {}
                })
            elif agent_name == 'highlighting_agent':
                frontend_actions.append({
                    'type': 'highlight_points',
                    'data': result.data or {}
                })
            elif agent_name == 'optimization_agent':
                if result.data and result.data.get('run_id'):
                    frontend_actions.append({
                        'type': 'start_optimization',
                        'data': result.data
                    })
            elif agent_name == 'report_agent':
                frontend_actions.append({
                    'type': 'report_generated',
                    'data': result.data or {}
                })
            elif agent_name == 'comparative_analysis_agent':
                if result.data:
                    frontend_actions.append({
                        'type': 'comparative_analysis_result',
                        'data': result.data
                    })
        bot.last_agent_results = []

        return Response({"response": response, "frontend_actions": frontend_actions})

    except Exception as e:
        print(f"Error in get_chat_response: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
    
    finally:
        if bot:
            bot.last_agent_results = []


@api_view(["GET"])
def get_latest_run_directory(request):
    """
    Return the most recent myrun_ directory for chat-triggered run discovery [1].
    """
    try:
        base_path = "api/Evaluator/cascade/chiplet_model/dse/results"
        if not os.path.exists(base_path):
            return Response({"error": "results directory not found"}, status=404)
        
        candidates = [d for d in os.listdir(base_path) if d.startswith("myrun_")]
        if not candidates:
            return Response({"status": "empty"})
        
        latest = sorted(candidates)[-1]
        return Response({"status": "success", "run_directory": latest})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(["POST"])
def add_insights_context(request):
    """
    Add plain insights string to chatbot history [1].
    """
    try:
        data = json.loads(request.body)
        insights = data.get("insights")
        
        if not insights:
            return Response({"error": "No insights provided"}, status=400)
        
        bot = get_chatbot()
        bot.messages.append({
            "role": "assistant",
            "content": f"Here are the insights from the analysis:\n\n{insights}\n\nI have this context and can answer follow-up questions about these insights."
        })
        return Response({"message": "Insights context added successfully"})
    except Exception as e:
        return Response({"error": str(e)}, status=500)