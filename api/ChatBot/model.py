import os
import dotenv
from openai import OpenAI
import json
import csv
import numpy as np
from dcor import distance_correlation
from collections import Counter
from copy import deepcopy

dotenv.load_dotenv()

class ChatBotModel():

    def __init__(self, specs=None, model: str="gpt-4o-mini"):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._call_data = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.set_specs(specs=specs)
        self.call_data_specs()
        self.messages = self._specs.copy()
        self.call_data_messages = self._call_data_specs.copy()
        self.call_data_context()
        self._model = model
        self.pointContext = {}
        self.pointInActiveContext = False
        self.get_pareto_front_questions()

    def call_data_specs(self):
        self._call_data_specs = [
            {
                "role": "developer",
                "content": "You are an expert in chiplet design and optimization. " +
                           "Users will come to you with vague questions about chiplet design, and your only job is to determine " + 
                           "what additional data is needed to correctly answer the question. " +
                           "To parse the dataset you need to specify a parameter ('flops', 'mem_accessed', 'exe_time', or 'energy'), " +
                           "whether you want the minimum ('min') or maximum ('max') values for that parameter, " +
                           "and the number of values you want to return. " +
                           "For energy bottleneck analysis, always request the maximum energy values to identify the highest consumers. " +
                           "For comprehensive energy analysis, consider requesting both energy and memory access data. " +
                           "Do not respond with anything except the data request."
            }
        ]

    def call_data_context(self):
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "What are the 3 minimum flops chiplets?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'min', 'param':'flops', 'num_points':3)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "What is the main execution time bottleneck of this design?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'exe_time', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "Investigate how large memory access relates to large energy usage."
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'mem_accessed', 'num_points':10)\n" +
                           "parse_dataset('min_max':'max', 'param':'energy', 'num_points':10)"
            }
        )
        # Enhanced energy bottleneck analysis examples
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "What is the bottleneck for energy?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'energy', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "Which chiplet consumes the most energy?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'energy', 'num_points':3)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "What are the energy bottlenecks in this design?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'energy', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "Show me the highest energy consuming chiplets"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'energy', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "What's causing high energy consumption?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'energy', 'num_points':5)\n" +
                           "parse_dataset('min_max':'max', 'param':'mem_accessed', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "Analyze energy distribution across chiplets"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'energy', 'num_points':12)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "Which chiplets are energy inefficient?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'energy', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "What's the energy profile of this design?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'energy', 'num_points':12)\n" +
                           "parse_dataset('min_max':'min', 'param':'energy', 'num_points':3)"
            }
        )
        # Enhanced runtime bottleneck analysis examples
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "What is the bottleneck for runtime?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'exe_time', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "Which chiplet is the slowest?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'exe_time', 'num_points':3)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "What are the runtime bottlenecks in this design?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'exe_time', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "Show me the slowest chiplets"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'exe_time', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "What's causing slow execution?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'exe_time', 'num_points':5)\n" +
                           "parse_dataset('min_max':'max', 'param':'flops', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "Analyze execution time distribution across chiplets"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'exe_time', 'num_points':12)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "Which chiplets are performance bottlenecks?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'exe_time', 'num_points':5)"
            }
        )
        self.call_data_messages.append(
            {
                "role": "user",
                "content": "What's the performance profile of this design?"
            }
        )
        self.call_data_messages.append(
            {
                "role": "assistant",
                "content": "parse_dataset('min_max':'max', 'param':'exe_time', 'num_points':12)\n" +
                           "parse_dataset('min_max':'min', 'param':'exe_time', 'num_points':3)"
            }
        )

    def get_pareto_front_questions(self):
        pfront_question_list = [
            "Is this design Pareto optimal?",
            "What is the rank of this solution?",
            "How good is this point?",
            "Where are the best designs?",
            "Why is this design not optimal?",
            "What are some common features of the Pareto front?",
            "Based on the existing Pareto front, can you propose a new design?",
            "Can you tell me about how design features relate to performance?",
            "Are there any general rules which imply a good design?"
        ]

        self.pareto_front_questions = {}
        for question in pfront_question_list:
            embedding = self.get_embedding(question)
            self.pareto_front_questions[question] = embedding

    def set_specs(self, specs):
        self._specs = [
            {
                "role": "developer",
                "content": "You are an expert in chiplet design and optimization. " +
                           "Designs are made up of twelve chiplets, and the different numbers of different kinds of chiplets " +
                           "give different performance characteristics. Designs are primarily evaluated based on " +
                           "performance and power consumption for a given trace. You are to help the user design a chiplet based on " +
                           "the information you are provided. " +
                           "When analyzing bottlenecks, provide direct, concise answers that immediately identify the main bottleneck with EXACT NUMBERS. " +
                           "For energy bottlenecks, focus ONLY on what is causing HIGHER energy consumption. " +
                           "You MUST include: " +
                           "1) The exact chiplet type that is the bottleneck (e.g., 'Convolution', 'GPU', 'Attention', 'Sparse') " +
                           "2) The EXACT average energy per chiplet in mJ (e.g., '1234.56 mJ per Convolution chiplet') showing why it's a high consumer " +
                           "3) The EXACT total energy contribution in mJ and percentage (e.g., '45678.90 mJ, 28.65% of total energy') showing its impact " +
                           "4) The specific kernels where this chiplet type appears as top consumer " +
                           "5) Comparison ONLY with other HIGH energy consumers, not low energy consumers " +
                           "CRITICAL: Focus exclusively on HIGH energy consumers. Do NOT mention low energy consumers. " +
                           "Use ONLY the exact numbers provided in the kernel breakdown for HIGH ENERGY CONSUMERS. " +
                           "NEVER say 'depends on detailed breakdown' or 'varies' or 'exact value depends on' - " +
                           "always state the specific values from the breakdown data. " +
                           "For runtime bottlenecks, focus on: " +
                           "1) Which chiplet type is the main performance bottleneck " +
                           "2) The exact runtime values (in ms) from the provided breakdown " +
                           "3) The percentage contribution to total runtime " +
                           "4) Brief explanation of why this is the bottleneck " +
                           "Keep responses focused and to the point, avoiding unnecessary verbosity. " +
                           "Be direct and actionable in your analysis."
            }
        ] if specs is None else specs

    def set_model(self, model: str):
        self._model = model

    def get_response_with_retrieval(self, content, role="user", filters=None, top_k: int = 6):
        """
        Retrieval-augmented answer with concise citations.
        """
        from api.retrieval import search

        # 1) Retrieve
        results = search(content, self.get_embedding, top_k=top_k, filters=filters)

        # 2) Build compact, token-bounded context
        context_lines = []
        citations = []
        for r in results:
            ctag = f"C{r['rank']}"
            text = r["text"]
            meta = r.get("metadata", {})
            # Keep each snippet brief
            snippet = text if len(text) <= 300 else text[:297] + "..."
            context_lines.append(f"[{ctag}] {snippet}")
            citations.append({
                "tag": ctag,
                "score": r.get("score"),
                "file_path": meta.get("file_path"),
                "metadata": meta
            })

        context_block = "\n".join(context_lines)

        # 3) Prompt with instructions to cite
        system_msg = {
            "role": "developer",
            "content": (
                "Answer using ONLY the provided context. Cite sources as [C#]. "
                "If context is insufficient, say what is missing and request it explicitly. "
                "Be concise and technical."
            )
        }

        user_msg = {
            "role": role,
            "content": f"Question: {content}\n\nContext:\n{context_block}"
        }

        messages = [system_msg] + self._specs + [user_msg]
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages
        )
        answer = completion.choices[0].message.content

        # Append to history minimally
        self.messages.append({"role": role, "content": content})
        self.messages.append({"role": "assistant", "content": answer})

        return {
            "final_answer": answer,
            "citations": citations
        }

    def get_response(self, content, role="user"):
        self.messages.append(
            {
                "role": role,
                "content": content
            }
        )

        # Check if this is an energy-related question
        energy_keywords = [
            'energy', 'bottleneck', 'consumption', 'efficient', 'power', 
            'consuming', 'bottlenecks', 'profile', 'distribution', 'inefficient'
        ]
        is_energy_question = any(keyword in content.lower() for keyword in energy_keywords)
        
        # If energy question and we have context, add condensed kernel breakdown
        if is_energy_question and self.pointInActiveContext:
            try:
                breakdown = self.extract_kernel_breakdown()
                if breakdown:
                    # Create concise summary for LLM
                    summary = self._format_breakdown_for_llm(breakdown)
                    # Replace the last message (user question) with enhanced version
                    self.messages[-1]['content'] = f"Kernel-wise Energy Breakdown:\n{summary}\n\nUser question: {content}"
                    print("Added condensed kernel breakdown for energy analysis")
            except Exception as e:
                print(f"Error adding kernel breakdown for energy question: {e}")
                import traceback
                traceback.print_exc()
                # Continue with normal flow if breakdown fails
        
        # Check if this is a runtime-related question
        runtime_keywords = [
            'runtime', 'execution', 'time', 'slow', 'fast', 'performance', 'speed',
            'bottleneck', 'bottlenecks', 'slowest', 'fastest', 'latency', 'throughput'
        ]
        is_runtime_question = any(keyword in content.lower() for keyword in runtime_keywords)

        # Gather information from the point context if the user is asking about a specific chiplet
        if self.pointInActiveContext:
            self.call_data_messages.append(
                {
                    "role": 'user',
                    "content": content
                }
            )
            call_data_completion = self._call_data.chat.completions.create(
                model=self._model,
                messages=self.call_data_messages
            )
            call_data_response = call_data_completion.choices[0].message.content
            for line in call_data_response.splitlines():
                call_data_message = self.retrieve_point_parser(line)
                self.messages.append(
                    {
                        "role": 'user',
                        "content": call_data_message
                    }
                )

            print("The data response worked!")
            
            # Add enhanced energy analysis for energy-related questions
            if is_energy_question:
                self.add_enhanced_energy_analysis()
            
            # Add enhanced runtime analysis for runtime-related questions
            if is_runtime_question:
                self.add_enhanced_runtime_analysis()

        
        # Add pareto context if the content is similar to any of the pareto front questions
        content_embedding = self.get_embedding(content)
        for question in self.pareto_front_questions:
            pareto_similarity = self.cosine_similarity(content_embedding, self.pareto_front_questions[question])
            print(f"Question: {question}")
            print(f"Pareto similarity: {pareto_similarity}")
            if pareto_similarity > 0.6:
                print("Adding pareto context.")
                self.add_pareto_context()
                break

        print("Message History: ", self.messages)
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=self.messages
        )
        print(completion.choices[0].message.content)
        self.messages.append(
            {
                "role": completion.choices[0].message.role,
                "content": completion.choices[0].message.content
            }
        )
        return completion.choices[0].message.content

    def add_information(self, contextFilePath):
        # self.pointContext = {} # reset to zero in case of multiple calls
        self.pointContext = []
        self.pointInActiveContext = True
        try:
            with open(contextFilePath, 'r') as file:
                self.full_data = json.load(file)
                for kernal in range(len(self.full_data)):
                    kernal_data = []
                    for chiplet in self.full_data[kernal]['chiplets']:
                        chiplet_data = []
                        for value in self.full_data[kernal]['chiplets'][chiplet]:
                            if not value == 'name':
                                chiplet_data.append(self.full_data[kernal]['chiplets'][chiplet][value])
                        kernal_data.append(chiplet_data)
                    total_data = []
                    for value in self.full_data[kernal]['total']:
                        total_data.append(self.full_data[kernal]['total'][value])
                    kernal_data.append(total_data)
                    self.pointContext.append(kernal_data)
                
                self.pointContext = np.array(self.pointContext)
                print(f"Point Context Shape: {self.pointContext.shape}")


        except FileNotFoundError:
            print(f"Error: File at {contextFilePath} not found.")
        # except Exception as e:
        #     print(f"An error occurred: {e}")

    def retrieve_point_parser(self, model_call):
        """
        model_call: 'parse_dataset('min_max':'min', 'param':'flops', 'num_points':5)'
        """
        # Parse the model call
        print(f"Model call: {model_call}")
        min_max_ind = [model_call.find("'min_max':'") + len("'min_max':'"), model_call.find("'param'") - 3]
        param_ind = [model_call.find("'param':'") + len("'param':'"), model_call.find("'num_points'") - 3]
        num_points_ind = [model_call.find("'num_points':") + len("'num_points':"), model_call.find(")")]
        min_max = model_call[min_max_ind[0]:min_max_ind[1]]
        param = model_call[param_ind[0]:param_ind[1]]
        num_points = int(model_call[num_points_ind[0]:num_points_ind[1]])
        return self.retrieve_point_data(min_max, param, num_points)

    def retrieve_point_data(self, min_max, param, num_points):
        """
        min_max: 'min' or 'max'
        chiplet:'0'-'11' or 'total'
        params: 'flops', 'mem_accessed', 'exe_time', or 'energy'
                'work' and 'energy_dram' are not supported yet
        num_points: number of values to return. For example, if num_points = 5 and 'max', return the 5 largest values for the param and chiplet
        """
        param_dict = {"flops": 0, "mem_accessed": 1, "exe_time": 2, "energy": 3}

        if min_max not in ['min', 'max']:
            raise ValueError("min_max must be 'min' or 'max'")
        if param == 'work' or param == 'energy_dram':
            raise ValueError("work and energy_dram are not supported yet")
        
        param_ind = param_dict[param]
        
        vals = self.pointContext[:, -1, param_ind]
        if min_max == 'min':
            ind = np.argpartition(vals, num_points)[:num_points]
        elif min_max == 'max':
            ind = np.argpartition(vals, -num_points)[-num_points:]

        chiplet = np.argmax(self.pointContext[ind, :-1, param_ind], axis=1)
        # print("Chiplet: ", chiplet)

        totals_string = ""
        for i in range(len(ind)):
            totals_string_add = (
                f"Kernal number {self.full_data[ind[i]]['kernal_number']}, a {self.full_data[ind[i]]['name']} type kernal, "
                f"has a total {param} of {vals[ind[i]]}. The {min_max} {param} chiplet is chiplet number {chiplet[i]} which is a "
                f"{self.full_data[ind[i]]['chiplets'][str(chiplet[i])]['name']} type chiplet and has a value of "
                f"{self.full_data[ind[i]]['chiplets'][str(chiplet[i])][param]}.\n"
            )
            totals_string += totals_string_add

        message = f"The {num_points} {min_max} {param} chiplets are:\n {totals_string}"
        print(f"Message: {message}")

        return message


    def add_pareto_context(self):
        file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            full_data = []
            for row in csv_reader:         
                full_data.append(
                    json.dumps(
                        {
                            "exe_time": float(row[0]),
                            "energy": float(row[1]),
                            "chiplets": {
                                "GPU": int(row[2]),
                                "Attention": int(row[3]),
                                "Sparse": int(row[4]),
                                "Convolution": int(row[5])
                            }
                        },
                        sort_keys=True
                    )
                )
        full_data = [json.loads(item) for item in set(full_data)]  # convert to set to remove duplicates, then back to list and parse JSON items to dict

        point_vals = []
        design_vals = []
        for data in full_data:
            point_vals.append([data["exe_time"], data["energy"]])
            design_vals.append([data["chiplets"]["GPU"], data["chiplets"]["Attention"],
                                data["chiplets"]["Sparse"], data["chiplets"]["Convolution"]])

        point_vals = np.array(point_vals)
        design_vals = np.array(design_vals)

        pfront_data = []
        prank = 0
        point_vals_copy = deepcopy(point_vals)
        print(f"Full data: {full_data}")
        # print(f"Point vals: {point_vals}")
        while len(point_vals_copy) > 0:
            pfront_mask = self.is_pareto_efficient(point_vals_copy)
            next_prank = point_vals_copy[pfront_mask]
            pfront_data.extend(next_prank)
            point_vals_copy = point_vals_copy[~pfront_mask]
            for data in full_data:
                if [data["exe_time"], data["energy"]] in next_prank:
                    data["rank"] = prank
            prank += 1

        print(f"Pareto front data: {full_data}")

        distance_correlation_str = self.get_distance_correlations(point_vals, design_vals)

        rule_mining_str = self.rule_mining()

        self.messages.append(
            {
                "role": "user",
                "content": "Here is some information about each point including their performance charactaristics " +
                           "(exe_time, energy), thier pareto ranking (rank), and the number of each kind of chiplet in the design " + 
                           "(GPU, Attention, Sparse, Convolution). The number of each kind of chiplet are the design variables, " +
                           "but there are always twelve chiplets in total. " + 
                           "When answering questions, keep your responses as concise as possible and do not restate things I already know. \n" +
                           f"{str(full_data)}\n" + 
                           "In addition, here are the distance correlation numbers between each objective and the number of each type of chiplet:\n" + 
                           distance_correlation_str + 
                           rule_mining_str
            }
        )


    def get_distance_correlations(self, objective_vals, design_vals):
        objectives = ["exe_time", "energy"]
        chiplets = ["GPU", "Attention", "Sparse", "Convolution"]
        dist_corr_str = ""

        for obj_ind, obj in enumerate(objectives):
            for chiplet_ind, chiplet in enumerate(chiplets):
                obj_vals = np.array(objective_vals[:, obj_ind], dtype=float)
                des_vals = np.array(design_vals[:, chiplet_ind], dtype=float)
                dist_corr = distance_correlation(obj_vals, des_vals)
                dist_corr_str = dist_corr_str + f"Distance correlation between objective {obj} and chiplet type {chiplet} is {dist_corr:.4f}.\n"

        print("Distance Correlation String: ", dist_corr_str, "\n\n")
        return dist_corr_str
    

    def rule_mining(self, point_selection_params=None):
        """
        Perform rule mining on the data to find patterns in the pareto front
        """
        # Get file path from parameters or use default
        file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        if point_selection_params and "file_path" in point_selection_params:
            file_path = point_selection_params["file_path"]
        
        print(f"[ChatBot] Reading data from: {file_path}")
        full_data = []
        with open(file_path, "r") as f:
            for line in f:
                row = line.strip().split(",")
                if len(row) >= 6:  # Ensure we have at least 6 columns
                    full_data.append(
                        (float(row[0]), float(row[1]), int(round(float(row[2]))), int(round(float(row[3]))), int(round(float(row[4]))), int(round(float(row[5]))))  # exe_time, energy, GPU, Attention, Sparse, Convolution
                    )
        
        # Check if we have any data
        if not full_data:
            print("[ChatBot] No data found in CSV file")
            return "No data available for rule mining analysis."
        
        full_data = np.array(list(set(full_data)))
        
        # Check if the array has the expected shape
        if len(full_data.shape) != 2 or full_data.shape[1] < 2:
            print(f"[ChatBot] Unexpected data shape: {full_data.shape}")
            return "Data format is not suitable for rule mining analysis."
        
        point_vals = full_data[:, :2]  # exe_time, energy
        max_vals = np.max(point_vals, axis=0)*1.1 + 1e-6  # add a small value to avoid division by zero

        prank = 0
        prank_array = np.zeros(len(full_data), dtype=int)  # to store the rank of each point
        while np.min(point_vals[:,0]) < max_vals[0]:
            pfront_mask = self.is_pareto_efficient(point_vals)
            point_vals[pfront_mask] = max_vals # remove the pareto front points from the point_vals
            prank_array[pfront_mask] = prank
            prank += 1
        full_data = np.hstack((full_data, prank_array.reshape(-1, 1)))

        # Handle point selection based on user parameters
        if point_selection_params is None:
            # Default behavior: get the first three ranks of the pareto front
            point_selection = full_data[full_data[:, -1] < 3]
        else:
            region = point_selection_params.get("region", "pareto")
            
            if region == "all":
                # Select all points
                point_selection = full_data
            elif region == "pareto":
                # Select Pareto front ranks
                pareto_start = point_selection_params.get("pareto_start_rank", 1)
                pareto_end = point_selection_params.get("pareto_end_rank", 3)
                point_selection = full_data[(full_data[:, -1] >= pareto_start - 1) & (full_data[:, -1] < pareto_end)]
            elif region == "custom":
                # Select custom region based on energy and time ranges
                energy_min = point_selection_params.get("energy_min")
                energy_max = point_selection_params.get("energy_max")
                time_min = point_selection_params.get("time_min")
                time_max = point_selection_params.get("time_max")
                
                mask = np.ones(len(full_data), dtype=bool)
                if energy_min is not None:
                    mask &= (full_data[:, 1] >= energy_min)  # energy is column 1
                if energy_max is not None:
                    mask &= (full_data[:, 1] <= energy_max)
                if time_min is not None:
                    mask &= (full_data[:, 0] >= time_min)    # time is column 0
                if time_max is not None:
                    mask &= (full_data[:, 0] <= time_max)
                
                point_selection = full_data[mask]
            else:
                # Fallback to default behavior
                point_selection = full_data[full_data[:, -1] < 3]

        print(f"Selected {len(point_selection)} points for rule mining analysis")

        # Check if we have enough points for analysis
        if len(point_selection) == 0:
            return "No points selected for rule mining analysis. Please check your selection criteria."

        feature_list = ["none", "low", "medium", "high", "very high"]
        chiplet_list = ["GPU", "Attention", "Sparse", "Convolution"]
        rules_dict = {}
        for feature in feature_list:
            for chip_ind, chiplet in enumerate(chiplet_list):
                if feature == "none":
                    rule_points = full_data[full_data[:, chip_ind + 2] == 0]  # +2 because first two columns are exe_time and energy
                elif feature == "low":
                    rule_points = full_data[(full_data[:, chip_ind + 2] >= 1) & (full_data[:, chip_ind + 2] <= 2)]
                elif feature == "medium":
                    rule_points = full_data[(full_data[:, chip_ind + 2] >= 3) & (full_data[:, chip_ind + 2] <= 5)]
                elif feature == "high":
                    rule_points = full_data[(full_data[:, chip_ind + 2] >= 6) & (full_data[:, chip_ind + 2] <= 8)]
                elif feature == "very high":
                    rule_points = full_data[full_data[:, chip_ind + 2] >= 9]
                rules_dict[f"{chiplet}_{feature}"] = rule_points
        pfront_rules = []
        pfront_costs = np.array([]).reshape(0, 2)  # to store the pareto front costs conf(f->p) and conf(p->f)
        pfront_lifts = np.array([]).reshape(0, 1)  # to store the lift of the rules
        base_rule = set()
        new_pfront_rules, new_pfront_costs, new_pfront_lifts = self.add_rules(rules_dict, point_selection, pfront_rules, pfront_costs, pfront_lifts, base_rule, full_data)

        # print(f"Number of rules found: {len(new_pfront_rules)}")

        rule_mining_str = "Rules were defined for each design as a certain range of each type of chiplet. The ranges are as follows:\n" + \
            "num = 0 (none), \n1 <= num <= 2 (low), \n3 <= num <= 5 (medium), \n6 <= num <= 8 (high), \nnum >= 9 (very high)\n\n" + \
            "The prevelance of these rules were compared to the first three ranks of the pareto front. Given is the best combinations of rules " + \
            "which have high conf(f->p) (which is the confidence that a combination of rules implies being in the first three ranks of the pareto front) " + \
            "and conf(p->f) (which is the confidence that being on the pareto front implies a combination of rules). Also included is the lift, which is " + \
            "a ratio of how often the rule combination and the pareto front coincide to how often they would if they were independant.\n\n"
        
        if len(new_pfront_rules) == 0:
            rule_mining_str += "No significant rules found in the current dataset."
        else:
            for i, rule_set in enumerate(new_pfront_rules):
                rule_str = " AND ".join(rule_set)
                rule_mining_str += f"Rule: {rule_str}, conf(f->p): {new_pfront_costs[i,0]}, conf(p->f): {new_pfront_costs[i,1]}, lift: {new_pfront_lifts[i]}\n\n"
                # print(f"Rule: {rule_str}\n Confidence(f->p): {new_pfront_costs[i,0]}, Confidence(p->f): {new_pfront_costs[i,1]}")

        print(f"Rule mining string: {rule_mining_str}")
        return rule_mining_str

    def add_rules(self, rules_dict, point_selection, pfront_rules, pfront_costs, pfront_lifts, base_rule, full_data):
        """
        Iteratively add rules where the combination of rules has
        high support and is on the pareto front of
        conf(f->p) and conf(p->f)
        """

        prev_pfront_rules = deepcopy(pfront_rules)

        if len(pfront_rules) == 0:
            new_rules, pfront_rules, pfront_costs, pfront_lifts = self.find_confidences(rules_dict, point_selection, pfront_rules, pfront_costs, pfront_lifts, base_rule, full_data)
        else:
            for rule_set in pfront_rules:
                new_rules, pfront_rules, pfront_costs, pfront_lifts = self.find_confidences(rules_dict, point_selection, pfront_rules, pfront_costs, pfront_lifts, rule_set, full_data)
        
        pfront_mask = self.is_pareto_efficient(-pfront_costs, return_mask=True) # minus because we want to maximize the confidence values
        pfront_costs = pfront_costs[pfront_mask]
        pfront_lifts = pfront_lifts[pfront_mask]
        new_pfront_rules = [pfront_rules[i] for i in range(len(pfront_rules)) if pfront_mask[i]]
        # print(f"New pareto front rules: {new_pfront_rules}")
        # print(f"New pareto front costs: {pfront_costs}\n")
        if new_pfront_rules == prev_pfront_rules:
            return new_pfront_rules, pfront_costs, pfront_lifts
        new_pfront_rules, pfront_costs, pfront_lifts = self.add_rules(rules_dict, point_selection, new_pfront_rules, pfront_costs, pfront_lifts, base_rule, full_data)
        return new_pfront_rules, pfront_costs, pfront_lifts
    
    def find_confidences(self, rules_dict, point_selection, pfront_rules, pfront_costs, pfront_lifts, base_rule, full_data):
        min_supp = 0.05
        new_rules = 0
            
        for rule in rules_dict:
            new_rule = deepcopy(base_rule)
            new_rule.add(rule)
            if new_rule not in pfront_rules: # dont calculate for duplicate rule sets
                rule_points_all = [set(map(tuple, rules_dict[rules])) for rules in new_rule]
                rule_point_set = np.array(list(set.intersection(*rule_points_all)))

                p_and_f = []
                for point in rule_point_set:
                    if any(np.equal(point, point_selection).all(axis=1)):  # check if point is in point_selection
                        p_and_f.append(point)
                len_p_and_f = len(p_and_f)
                if len_p_and_f == 0:
                    continue
                # check if support is high enough
                rule_support = len_p_and_f / len(full_data)
                if rule_support >= min_supp:
                    # get the confidence of p to f and f to p
                    conf_p_to_f = len_p_and_f / len(point_selection)
                    conf_f_to_p = len_p_and_f / len(rule_point_set)

                    pfront_costs = np.vstack((pfront_costs, np.array([conf_p_to_f, conf_f_to_p])))
                    pfront_rules.append(new_rule)
                    pfront_lifts = np.vstack((pfront_lifts, len_p_and_f * len(full_data)/ (len(point_selection) * len(rule_point_set))))

                    new_rules += 1

        return new_rules, pfront_rules, pfront_costs, pfront_lifts

    def is_pareto_efficient(self, costs, return_mask = True):
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index<len(costs):
            nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype = bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    def get_embedding(self, content):
        embedding = self._client.embeddings.create(
            model='text-embedding-3-small',
            input=content
        )
        return embedding.data[0].embedding

    def cosine_similarity(self, a, b):
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(y ** 2 for y in b) ** 0.5
        cos_sim =  dot_product / (norm_a * norm_b) if (norm_a and norm_b) else 0
        return cos_sim

    def clear_history(self):
        self.messages = []
        self.messages = self._specs.copy()
        self.pointContext = {}
        self.pointInActiveContext = False

                # if self.pointInActiveContext:
        #     # Check if the content is similar to any point in the context
        #     relevant_context = ''
        #     # content_embedding = self.get_embedding(content)
        #     rewritten_content = self._adj_input.chat.completions.create(
        #         model=self._model,
        #         messages=self._adj_input_specs + [{"role": "user", "content": content}]
        #     )
        #     rewritten_message = rewritten_content.choices[0].message.content
        #     rewritten_content_embedding = self.get_embedding(rewritten_message)
        #     print(f"Rewritten message: {rewritten_message}")
        #     print(f"Length of pointContext: {len(self.pointContext)}")
        #     rel_context = {}
        #     for point_str in self.pointContext:
        #         similarity = self.cosine_similarity(rewritten_content_embedding, self.pointContext[point_str])
        #         # print(f"Similarity: {similarity}")
        #         if len(rel_context) < 5:
        #             rel_context[point_str] = similarity
        #         elif similarity > min(rel_context.values()):
        #             min_key = min(rel_context, key=rel_context.get)
        #             del rel_context[min_key]
        #             rel_context[point_str] = similarity
        #         # if similarity > 0.8:
        #         #     relevant_context = relevant_context + point_str + '\n'
        #     print(f"Length of rel_context: {len(rel_context)}")
        #     for point_str in rel_context:
        #         relevant_context += point_str + '\n'
        # if self.pointInActiveContext:
        #     self.messages.append(
        #         {
        #             "role": role,
        #             # "content": rewritten_message
        #             "content": content
        #         }
        #     )
        # else:
        #     self.messages.append(
        #         {
        #             "role": role,
        #             "content": content
        #         }
        #     )
        # if self.pointInActiveContext and relevant_context:
        #     print("Relevant context found.")
        #     self.messages.append(
        #         {
        #             "role": "user",
        #             "content": "Here are some of the most relevant kernals. Here 'name' gives the kernal type, 'flops' is floating-point operations per second, " +
        #                        "'mem_accessed' is the amount of memory accessed, 'exe_time' is the execution time, 'energy' is the energy used, " +
        #                        "and 'energy_dram' relates to the dynamic random access memory. When using this context, " +
        #                        "make your responses specific to the kernal and chiplet, and be as consise as possible while still providing all the relevant information." +
        #                        f"\n\n {str(relevant_context)}"
        #         }
        #     )

                        # batchSize = 90 # maximum size for this embedding model is 8192, batch size of 90 puts it just under that
                # data = json.load(file)
                # for i in range(0, len(data), batchSize):
                #     if i + batchSize > len(data):
                #         batch = data[i:]
                #     else:
                #         batch = data[i:i+batchSize]
                #     for dict in batch:
                #         if 'chiplets' in dict:
                #             del dict['chiplets']
                #     # if i == 0:
                #     #     print(f"Batch: {batch}")
                #     batch_str = json.dumps(batch)
                #     embedding = self.get_embedding(batch_str)
                #     self.pointContext[batch_str] = embedding
                #     print("Got embedding for an entry")
                # lines = file.readlines()
                # print(f"What the length of the embedding should be: {len(lines)/batchSize}")
                # for i in range(0, len(lines), batchSize):
                #     if i + batchSize > len(lines):
                #         batch = lines[i:]
                #     else:
                #         batch = lines[i:i+batchSize]
                #     batchStr = '\n'.join(batch)
                #     embedding = self.get_embedding(batchStr)
                #     self.pointContext[batchStr] = embedding
                #     print("Got embedding")

                    # def adj_input_specs(self):
    #     # self.adj_input_specs = [
    #     #     {
    #     #         "role": "developer",
    #     #         "content": "You are an expert in chiplet design and optimization, python, and data retrieval. " +
    #     #                    "You have access to a large dataset of chiplet designs, and your job is to take a user's question " +
    #     #                    "and identify what additional data is needed to correctly answer the question. " +
    #     #                    "The dataset is separated by the design, which is just different amounts of the different kinds of chiplets: " +
    #     #                    "GPU, Attention, Sparse, and Convolutional. For each kernal in the trace, you have access to the flops (floating-point operations per second), " +
    #     #                    "mem_accessed (amount of memory accessed), exe_time (execution time), energy (energy used), and energy_dram (dynamic random access memory) " +
    #     #                    "for the deign as a whole, and for each individual chiplet in the design (except replace energy_dram with work). " +
    #     #     }
    #     # ]
    #     self._adj_input_specs = [
    #         {
    #             "role": "developer",
    #             "content": "You are an expert in chiplet design and optimization. " +
    #                        "Users will come to you with vague questions about chiplet design, and your only job is to rewrite thier questions to be more specific, " +
    #                        "so that the questions can be closely matched to an existing dataset. One example line of the dataset is: " +
    #                        "\n {'name': 'Attention', 'chiplets': {0: {'flops': np.float64(7305919517376.859), 'mem_accessed': np.float64(581855534.4761195), 'exe_time': np.float64(0.24353065057922865), 'energy': np.float64(2.631951106194661), 'work': np.float64(0.15820702601508768), 'name': 'atten'}, 1: {'flops': np.float64(7305919517376.859), 'mem_accessed': np.float64(581855534.4761195), 'exe_time': np.float64(0.24353065057922865), 'energy': np.float64(2.631951106194661), 'work': np.float64(0.15820702601508768), 'name': 'atten'}, 2: {'flops': np.float64(7305919517376.859), 'mem_accessed': np.float64(581855534.4761195), 'exe_time': np.float64(0.24353065057922865), 'energy': np.float64(2.631951106194661), 'work': np.float64(0.15820702601508768), 'name': 'atten'}, 3: {'flops': np.float64(7305919517376.859), 'mem_accessed': np.float64(581855534.4761195), 'exe_time': np.float64(0.24353065057922865), 'energy': np.float64(2.631951106194661), 'work': np.float64(0.15820702601508768), 'name': 'atten'}, 4: {'flops': np.float64(7305919517376.859), 'mem_accessed': np.float64(581855534.4761195), 'exe_time': np.float64(0.24353065057922865), 'energy': np.float64(2.631951106194661), 'work': np.float64(0.15820702601508768), 'name': 'atten'}, 5: {'flops': np.float64(244301057223.21262), 'mem_accessed': np.float64(553942701.0910256), 'exe_time': np.float64(0.2443010572232126), 'energy': np.float64(0.6967985892026394), 'work': np.float64(0.005290250409096115), 'name': 'sparse'}, 6: {'flops': np.float64(244301057223.21262), 'mem_accessed': np.float64(553942701.0910256), 'exe_time': np.float64(0.2443010572232126), 'energy': np.float64(0.6967985892026394), 'work': np.float64(0.005290250409096115), 'name': 'sparse'}, 7: {'flops': np.float64(1832257733052.2563), 'mem_accessed': np.float64(560219501.5674703), 'exe_time': np.float64(0.24430103107363418), 'energy': np.float64(1.7582047001505203), 'work': np.float64(0.03967687382127389), 'name': 'conv'}, 8: {'flops': np.float64(1832257733052.2563), 'mem_accessed': np.float64(560219501.5674703), 'exe_time': np.float64(0.24430103107363418), 'energy': np.float64(1.7582047001505203), 'work': np.float64(0.03967687382127389), 'name': 'conv'}, 9: {'flops': np.float64(1832257733052.2563), 'mem_accessed': np.float64(560219501.5674703), 'exe_time': np.float64(0.24430103107363418), 'energy': np.float64(1.7582047001505203), 'work': np.float64(0.03967687382127389), 'name': 'conv'}, 10: {'flops': np.float64(1832257733052.2563), 'mem_accessed': np.float64(560219501.5674703), 'exe_time': np.float64(0.24430103107363418), 'energy': np.float64(1.7582047001505203), 'work': np.float64(0.03967687382127389), 'name': 'conv'}, 11: {'flops': np.float64(1832257733052.2563), 'mem_accessed': np.float64(560219501.5674703), 'exe_time': np.float64(0.24430103107363418), 'energy': np.float64(1.7582047001505203), 'work': np.float64(0.03967687382127389), 'name': 'conv'}}, 'total': {'flops': np.float64(46179488366592.01), 'mem_accessed': np.float64(735513149.44), 'exe_time': np.float64(0.2443010572232126), 'energy': np.float64(23.34437621013119), 'energy_dram': np.float64(1.9808869311655408)}}" +
    #                        "\n where 'flops' is floating-point operations per second, 'mem_accessed' is the amount of memory accessed, " +
    #                        "'exe_time' is the execution time, 'energy' is the energy used, and 'energy_dram' relates to the dynamic random access memory. " +
    #                        "For example, if the user asks 'what is the main bottleneck of this design in terms of execution time?', you might rewrite it to: " + 
    #                        "Where in this design is 'flops' low and 'exe_time' high? " +
    #                        "When rewriting the user's question, make it as specific as possible to match existing dataset entries. " +
    #                        "Keep your responses as consise as possible while still providing all the relevant information, " +
    #                        "as if you are talking to another expert in the field."
    #         }
    #     ]

    def handle_natural_language_optimization(self, content):
        import re
        result = {
            "model": None,
            "algorithm": None,
            "traces": [],
            "objectives": [],
            "population_size": 0,
            "generations": 0
        }

        # 1. Extract model and algorithm
        model_match = re.search(r"\bmodel\s+(CASCADE|HISIM)\b", content, re.IGNORECASE)
        algo_match = re.search(r"\balgorithm\s+(Genetic Algorithm|Full-Factorial)\b", content, re.IGNORECASE)
        if model_match:
            result["model"] = model_match.group(1).upper()
        if algo_match:
            result["algorithm"] = algo_match.group(1)

        # 2. Extract population and generations
        pop_match = re.search(r"\bpopulation\s+(\d+)", content, re.IGNORECASE)
        gen_match = re.search(r"\bgenerations?\s+(\d+)", content, re.IGNORECASE)
        if pop_match:
            result["population_size"] = int(pop_match.group(1))
        if gen_match:
            result["generations"] = int(gen_match.group(1))

        # 3. Extract objectives
        objectives = re.findall(r"\b(minimize|optimize for)\s+(energy|time|latency)", content, re.IGNORECASE)
        if objectives:
            result["objectives"] = list(set([obj[1].capitalize() for obj in objectives]))

        # 4. Extract trace names
        trace_matches = re.findall(r"\btrace\s+(gpt-[a-z0-9\-]+)", content, re.IGNORECASE)
        for trace in trace_matches:
            result["traces"].append({ "name": trace })

        # Basic validation
        if not (result["model"] and result["algorithm"] and result["traces"] and result["objectives"]):
            return { "status": "error", "message": "Could not extract full optimization setup. Please specify model, algorithm, trace, and objectives." }

        # Call the GA run
        from api.Evaluator.gaCascade import runGACascade
        from api.Evaluator.generator import generate_weighted_trace
        from api.views import convert_ndarrays

        trace_input = result["traces"]
        if len(trace_input) > 1:
            composite = generate_weighted_trace(trace_input)
            trace_used = composite
        else:
            trace_used = trace_input[0]['name']

        ga_result = runGACascade(
            pop_size=result["population_size"],
            n_gen=result["generations"],
            trace=trace_used
        )
        ga_result = convert_ndarrays(ga_result)

        # Save results in memory for now
        self.latest_result = ga_result
        return {
            "status": "success",
            "message": f"Optimization run completed using {result['algorithm']} on {trace_used}",
            "data": ga_result
        }

    def get_data_mining_followup_response(self, question, data_mining_type, structured_data):
        """
        Handle follow-up questions about data mining results with context-aware responses.
        
        Args:
            question (str): User's follow-up question
            data_mining_type (str): "rule_mining" or "distance_correlation"
            structured_data (dict): The structured data from the analysis
        """
        # Create context-specific prompt based on data mining type
        if data_mining_type == "distance_correlation":
            context_prompt = (
                f"You are analyzing distance correlation results for chiplet design optimization.\n\n"
                f"Trace: {structured_data.get('trace_name', 'Unknown')}\n"
                f"Objective: {structured_data.get('objective', 'both')}\n\n"
                f"Correlation Data:\n"
                f"High impact on energy: {structured_data.get('high_impact_on_energy', [])}\n"
                f"High impact on time: {structured_data.get('high_impact_on_time', [])}\n\n"
                f"User Question: {question}\n\n"
                f"Provide a focused, technical answer based on the correlation data. "
                f"Explain the relationships between chiplet configurations and performance metrics."
            )
        elif data_mining_type == "rule_mining":
            context_prompt = (
                f"You are analyzing rule mining results for chiplet design optimization.\n\n"
                f"Trace: {structured_data.get('trace_name', 'Unknown')}\n"
                f"Objective: {structured_data.get('objective', 'both')}\n"
                f"Analysis Region: {structured_data.get('analysis_region', 'Unknown')}\n\n"
                f"Rules Found:\n{json.dumps(structured_data.get('rules', []), indent=2)}\n\n"
                f"User Question: {question}\n\n"
                f"Provide a focused, technical answer based on the rule patterns. "
                f"Explain the confidence levels, lift values, and what they mean for design decisions."
            )
        else:
            context_prompt = (
                f"You are analyzing {data_mining_type} results for chiplet design optimization.\n\n"
                f"Data: {json.dumps(structured_data, indent=2)}\n\n"
                f"User Question: {question}\n\n"
                f"Provide a focused, technical answer based on the data."
            )
        
        return self.get_response(context_prompt, role="user")

    def add_run_context(self, summary_text, analytics_text=None, suggestions=None):
        """
        Unified method to add run context, analytics insights, and follow-up suggestions to ChatBot.
        
        Args:
            summary_text (str): Run completion summary
            analytics_text (str, optional): Analytics insights summary
            suggestions (str, optional): Follow-up question suggestions
        """
        # Add run summary
        self.messages.append({"role": "assistant", "content": summary_text})
        
        # Add analytics insights if provided
        if analytics_text:
            self.messages.append({"role": "assistant", "content": analytics_text})
        
        # Add follow-up suggestions if provided
        if suggestions:
            self.messages.append({"role": "assistant", "content": suggestions})
        
        print(f"ChatBot: Added run context - Summary: {len(summary_text)} chars, Analytics: {len(analytics_text) if analytics_text else 0} chars, Suggestions: {len(suggestions) if suggestions else 0} chars")

    def extract_kernel_breakdown(self, context_file_path=None, output_format='json'):
        """
        Extract condensed kernel-wise breakdown of energy and runtime from context file.
        
        Args:
            context_file_path: Path to context JSON file. If None, uses self.full_data
            output_format: 'json' or 'csv' for output format
            
        Returns:
            Dictionary with kernel breakdown data, or path to saved file
        """
        # Load data if file path provided
        if context_file_path:
            try:
                with open(context_file_path, 'r') as f:
                    kernel_data = json.load(f)
            except Exception as e:
                print(f"Error loading context file: {e}")
                return None
        elif hasattr(self, 'full_data') and len(self.full_data) > 0:
            kernel_data = self.full_data
        else:
            print("No kernel data available for extraction")
            return None
        
        # Extract condensed breakdown
        breakdown = {
            'total_energy_mj': 0,
            'total_runtime_ms': 0,
            'kernels': []
        }
        
        # Process each kernel
        for kernel in kernel_data:
            kernel_name = kernel.get('name', 'Unknown')
            kernel_total = kernel.get('total', {})
            kernel_energy = kernel_total.get('energy', 0) * 1000  # Convert to mJ
            kernel_runtime = kernel_total.get('exe_time', 0) * 1000  # Convert to ms
            
            # Get chiplet breakdown - top 5 energy consumers per kernel
            chiplets = kernel.get('chiplets', {})
            chiplet_breakdown = []
            
            for chiplet_id, chiplet_data in chiplets.items():
                chiplet_energy = chiplet_data.get('energy', 0) * 1000  # Convert to mJ
                chiplet_runtime = chiplet_data.get('exe_time', 0) * 1000  # Convert to ms
                chiplet_work = chiplet_data.get('work', 0)
                chiplet_name = chiplet_data.get('name', 'Unknown')
                
                chiplet_breakdown.append({
                    'chiplet_id': str(chiplet_id),
                    'chiplet_type': chiplet_name,
                    'energy_mj': round(chiplet_energy, 3),
                    'runtime_ms': round(chiplet_runtime, 3),
                    'work': chiplet_work
                })
            
            # Sort by energy (descending) and take top 5
            chiplet_breakdown.sort(key=lambda x: x['energy_mj'], reverse=True)
            top_chiplets = chiplet_breakdown[:5]
            
            # Calculate energy percentage for top chiplets
            for chiplet in top_chiplets:
                if kernel_energy > 0:
                    chiplet['energy_percentage'] = round((chiplet['energy_mj'] / kernel_energy) * 100, 2)
                else:
                    chiplet['energy_percentage'] = 0
            
            breakdown['kernels'].append({
                'kernel_name': kernel_name,
                'kernel_energy_mj': round(kernel_energy, 3),
                'kernel_runtime_ms': round(kernel_runtime, 3),
                'energy_percentage': 0,  # Will calculate after total
                'top_chiplets': top_chiplets,
                'num_chiplets_involved': len(chiplets)
            })
            
            breakdown['total_energy_mj'] += kernel_energy
            breakdown['total_runtime_ms'] += kernel_runtime
        
        # Calculate energy percentages for each kernel
        if breakdown['total_energy_mj'] > 0:
            for kernel in breakdown['kernels']:
                kernel['energy_percentage'] = round(
                    (kernel['kernel_energy_mj'] / breakdown['total_energy_mj']) * 100, 2
                )
        
        breakdown['total_energy_mj'] = round(breakdown['total_energy_mj'], 3)
        breakdown['total_runtime_ms'] = round(breakdown['total_runtime_ms'], 3)
        
        # Sort kernels by energy (descending)
        breakdown['kernels'].sort(key=lambda x: x['kernel_energy_mj'], reverse=True)
        
        # Identify top energy-consuming kernels (top 10)
        breakdown['top_energy_kernels'] = breakdown['kernels'][:10]
        
        # Identify bottleneck chiplet type across all kernels
        # IMPORTANT: We need to calculate totals from ALL chiplets, not just top 5
        # So we need to go back to the original kernel_data to get complete totals
        chiplet_type_totals = {}
        
        # Re-process kernel_data to get ALL chiplet energy (not just top 5)
        for kernel in kernel_data:
            kernel_name = kernel.get('name', 'Unknown')
            chiplets = kernel.get('chiplets', {})
            
            for chiplet_id, chiplet_data in chiplets.items():
                chiplet_energy = chiplet_data.get('energy', 0) * 1000  # Convert to mJ
                chiplet_type = chiplet_data.get('name', 'Unknown')
                
                if chiplet_type not in chiplet_type_totals:
                    chiplet_type_totals[chiplet_type] = {
                        'total_energy_mj': 0,
                        'count': 0,
                        'kernels': []
                    }
                
                chiplet_type_totals[chiplet_type]['total_energy_mj'] += chiplet_energy
                chiplet_type_totals[chiplet_type]['count'] += 1
                if kernel_name not in chiplet_type_totals[chiplet_type]['kernels']:
                    chiplet_type_totals[chiplet_type]['kernels'].append(kernel_name)
        
        # Calculate averages and find bottleneck
        bottleneck_type = None
        max_avg_energy = 0
        for chiplet_type, stats in chiplet_type_totals.items():
            stats['avg_energy_mj'] = round(stats['total_energy_mj'] / stats['count'], 3) if stats['count'] > 0 else 0
            if stats['avg_energy_mj'] > max_avg_energy:
                max_avg_energy = stats['avg_energy_mj']
                bottleneck_type = chiplet_type
        
        breakdown['chiplet_type_summary'] = chiplet_type_totals
        breakdown['energy_bottleneck'] = {
            'chiplet_type': bottleneck_type,
            'avg_energy_mj': max_avg_energy,
            'total_energy_mj': chiplet_type_totals.get(bottleneck_type, {}).get('total_energy_mj', 0),
            'kernels_affected': chiplet_type_totals.get(bottleneck_type, {}).get('kernels', [])
        }
        
        # Save to file if requested
        if output_format == 'csv':
            return self._save_breakdown_csv(breakdown, context_file_path)
        elif output_format == 'json' and context_file_path:
            return self._save_breakdown_json(breakdown, context_file_path)
        
        return breakdown
    
    def _save_breakdown_csv(self, breakdown, context_file_path=None):
        """Save kernel breakdown to CSV file"""
        import os
        from datetime import datetime
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analytics')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if context_file_path:
            design_name = os.path.basename(context_file_path).replace('.json', '')
            csv_path = os.path.join(output_dir, f"kernel_breakdown_{design_name}_{timestamp}.csv")
        else:
            csv_path = os.path.join(output_dir, f"kernel_breakdown_{timestamp}.csv")
        
        # Write CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['Kernel Name', 'Kernel Energy (mJ)', 'Kernel Runtime (ms)', 
                           'Energy %', 'Top Chiplet Type', 'Top Chiplet Energy (mJ)', 
                           'Top Chiplet Energy %', 'Num Chiplets'])
            
            # Write kernel data
            for kernel in breakdown['kernels']:
                top_chiplet = kernel['top_chiplets'][0] if kernel['top_chiplets'] else {}
                writer.writerow([
                    kernel['kernel_name'],
                    kernel['kernel_energy_mj'],
                    kernel['kernel_runtime_ms'],
                    kernel['energy_percentage'],
                    top_chiplet.get('chiplet_type', 'N/A'),
                    top_chiplet.get('energy_mj', 0),
                    top_chiplet.get('energy_percentage', 0),
                    kernel['num_chiplets_involved']
                ])
            
            # Summary row
            writer.writerow([])
            writer.writerow(['SUMMARY'])
            writer.writerow(['Total Energy (mJ)', breakdown['total_energy_mj']])
            writer.writerow(['Total Runtime (ms)', breakdown['total_runtime_ms']])
            writer.writerow(['Energy Bottleneck', breakdown['energy_bottleneck']['chiplet_type']])
            writer.writerow(['Bottleneck Avg Energy (mJ)', breakdown['energy_bottleneck']['avg_energy_mj']])
        
        print(f"Kernel breakdown saved to CSV: {csv_path}")
        return csv_path
    
    def _save_breakdown_json(self, breakdown, context_file_path=None):
        """Save kernel breakdown to JSON file"""
        import os
        from datetime import datetime
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'analytics')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if context_file_path:
            design_name = os.path.basename(context_file_path).replace('.json', '')
            json_path = os.path.join(output_dir, f"kernel_breakdown_{design_name}_{timestamp}.json")
        else:
            json_path = os.path.join(output_dir, f"kernel_breakdown_{timestamp}.json")
        
        # Write JSON
        with open(json_path, 'w') as f:
            json.dump(breakdown, f, indent=2)
        
        print(f"Kernel breakdown saved to JSON: {json_path}")
        return json_path
    
    def _format_breakdown_for_llm(self, breakdown):
        """Format kernel breakdown as concise text for LLM context with specific numbers"""
        summary = f"""KERNEL-WISE ENERGY BREAKDOWN ANALYSIS - HIGH ENERGY CONSUMPTION FOCUS
Total Energy: {breakdown['total_energy_mj']} mJ | Total Runtime: {breakdown['total_runtime_ms']} ms

TOP HIGH ENERGY-CONSUMING KERNELS (sorted by energy, showing what causes higher consumption):
"""
        for i, kernel in enumerate(breakdown['top_energy_kernels'][:10], 1):
            top_chiplet = kernel['top_chiplets'][0] if kernel['top_chiplets'] else {}
            summary += f"{i}. Kernel: '{kernel['kernel_name']}'\n"
            summary += f"   - Kernel Energy: {kernel['kernel_energy_mj']} mJ ({kernel['energy_percentage']}% of total)\n"
            summary += f"   - Kernel Runtime: {kernel['kernel_runtime_ms']} ms\n"
            summary += f"   - Top Energy Chiplet: {top_chiplet.get('chiplet_type', 'N/A')} ({top_chiplet.get('energy_mj', 0)} mJ, {top_chiplet.get('energy_percentage', 0)}% of kernel)\n"
            summary += f"   - Chiplets Involved: {kernel['num_chiplets_involved']}\n\n"
        
        # Energy bottleneck with exact numbers - focusing on high energy consumption
        bottleneck = breakdown['energy_bottleneck']
        summary += f"ENERGY BOTTLENECK IDENTIFICATION (Highest Energy Consumer):\n"
        summary += f"Primary Bottleneck Chiplet Type: {bottleneck['chiplet_type']}\n"
        summary += f"- Average Energy per {bottleneck['chiplet_type']} chiplet: {bottleneck['avg_energy_mj']} mJ\n"
        summary += f"- Total Energy from {bottleneck['chiplet_type']} chiplets: {bottleneck['total_energy_mj']} mJ\n"
        summary += f"- Percentage of total energy: {(bottleneck['total_energy_mj'] / breakdown['total_energy_mj'] * 100) if breakdown['total_energy_mj'] > 0 else 0:.2f}%\n"
        summary += f"- Number of kernels affected: {len(bottleneck['kernels_affected'])}\n"
        summary += f"- Affected kernels: {', '.join(bottleneck['kernels_affected'][:10])}\n\n"
        
        # Calculate average energy across all chiplet types to identify high consumers
        all_avg_energies = [stats['avg_energy_mj'] for stats in breakdown['chiplet_type_summary'].values()]
        avg_energy_threshold = sum(all_avg_energies) / len(all_avg_energies) if all_avg_energies else 0
        
        # Focus ONLY on chiplet types causing HIGHER energy consumption
        # Show top energy consumers (above average or top 3-5 highest)
        sorted_types = sorted(breakdown['chiplet_type_summary'].items(), 
                             key=lambda x: x[1]['avg_energy_mj'], reverse=True)
        
        # Get top high energy consumers (at least top 3, or those above average threshold)
        high_energy_types = []
        for chiplet_type, stats in sorted_types:
            if len(high_energy_types) < 3 or stats['avg_energy_mj'] >= avg_energy_threshold:
                high_energy_types.append((chiplet_type, stats))
        
        summary += "HIGH ENERGY CONSUMERS (Chiplet Types Causing Higher Energy Consumption):\n"
        summary += "Focus ONLY on these chiplet types that are causing higher energy consumption:\n\n"
        for chiplet_type, stats in high_energy_types:
            pct = (stats['total_energy_mj'] / breakdown['total_energy_mj'] * 100) if breakdown['total_energy_mj'] > 0 else 0
            summary += f"{chiplet_type}:\n"
            summary += f"  - Average energy per chiplet: {stats['avg_energy_mj']} mJ\n"
            summary += f"  - Total energy: {stats['total_energy_mj']} mJ ({pct:.2f}% of total)\n"
            summary += f"  - Number of instances: {stats['count']}\n"
            summary += f"  - Present in kernels: {', '.join(stats['kernels'][:5])}{'...' if len(stats['kernels']) > 5 else ''}\n\n"
        
        summary += "INSTRUCTIONS FOR ANALYSIS:\n"
        summary += "When answering energy bottleneck questions, focus ONLY on what is causing HIGHER energy consumption. "
        summary += "Use the EXACT numbers provided above for HIGH ENERGY CONSUMERS only. "
        summary += "Be specific about:\n"
        summary += "1. The exact chiplet type that is the bottleneck (from the HIGH ENERGY CONSUMERS list above)\n"
        summary += "2. The exact average energy value (in mJ) showing why it's a high consumer\n"
        summary += "3. The exact total energy contribution (in mJ and percentage) showing its impact\n"
        summary += "4. The specific kernels where this chiplet type appears as the top consumer\n"
        summary += "5. Compare the bottleneck chiplet's energy values ONLY with other HIGH ENERGY CONSUMERS from the list above.\n"
        summary += "DO NOT mention low energy consumers - focus exclusively on what causes higher energy consumption.\n"
        
        return summary
    
    def add_enhanced_energy_analysis(self):
        """
        Add comprehensive energy analysis context to help with energy bottleneck questions.
        """
        if not self.pointInActiveContext or len(self.full_data) == 0:
            return
        
        # Analyze energy distribution across all chiplets
        energy_analysis = []
        for kernel_idx, kernel_data in enumerate(self.full_data):
            kernel_name = kernel_data.get('name', f'Kernel {kernel_idx}')
            chiplets = kernel_data.get('chiplets', {})
            
            # Sort chiplets by energy consumption
            sorted_chiplets = sorted(
                chiplets.items(), 
                key=lambda x: x[1].get('energy', 0), 
                reverse=True
            )
            
            # Get top energy consumers
            top_consumers = sorted_chiplets[:3]  # Focus on top 3
            
            # Calculate energy statistics
            all_energies = [chiplet[1].get('energy', 0) for chiplet in sorted_chiplets]
            total_energy = sum(all_energies)
            
            # Group by chiplet type
            type_energy = {}
            for chiplet_id, chiplet_data in sorted_chiplets:
                chiplet_type = chiplet_data.get('name', 'unknown')
                if chiplet_type not in type_energy:
                    type_energy[chiplet_type] = []
                type_energy[chiplet_type].append(chiplet_data.get('energy', 0))
            
            # Calculate type-specific statistics
            type_stats = {}
            for chiplet_type, energies in type_energy.items():
                type_stats[chiplet_type] = {
                    'count': len(energies),
                    'total_energy': sum(energies),
                    'avg_energy': sum(energies) / len(energies),
                    'percentage': (sum(energies) / total_energy) * 100
                }
            
            energy_analysis.append({
                'kernel_name': kernel_name,
                'kernel_number': kernel_data.get('kernal_number', kernel_idx),
                'total_energy': total_energy,
                'top_consumers': top_consumers,
                'type_statistics': type_stats
            })
        
        # Create concise energy analysis message focusing on HIGH energy consumption
        analysis_message = "Energy Bottleneck Analysis - High Energy Consumers:\n\n"
        
        # Identify the main bottleneck across all kernels
        all_type_stats = {}
        for analysis in energy_analysis:
            for chiplet_type, stats in analysis['type_statistics'].items():
                if chiplet_type not in all_type_stats:
                    all_type_stats[chiplet_type] = {
                        'total_energy': 0,
                        'count': 0,
                        'avg_energy': 0
                    }
                all_type_stats[chiplet_type]['total_energy'] += stats['total_energy']
                all_type_stats[chiplet_type]['count'] += stats['count']
        
        # Calculate overall averages
        for chiplet_type in all_type_stats:
            all_type_stats[chiplet_type]['avg_energy'] = (
                all_type_stats[chiplet_type]['total_energy'] / 
                all_type_stats[chiplet_type]['count']
            )
        
        # Calculate threshold for high energy consumers (above average)
        all_avg_energies = [stats['avg_energy'] for stats in all_type_stats.values()]
        avg_threshold = sum(all_avg_energies) / len(all_avg_energies) if all_avg_energies else 0
        
        # Find the main bottleneck (highest energy consumer)
        main_bottleneck = max(all_type_stats.items(), key=lambda x: x[1]['avg_energy'])
        bottleneck_type = main_bottleneck[0]
        bottleneck_energy = main_bottleneck[1]['avg_energy']
        
        # Get supporting evidence from kernels
        supporting_kernels = []
        for analysis in energy_analysis:
            if analysis['type_statistics'].get(bottleneck_type, {}).get('avg_energy', 0) > 0:
                supporting_kernels.append(analysis['kernel_name'])
        
        analysis_message += f"Main Energy Bottleneck (Highest Consumer): {bottleneck_type.upper()} chiplets\n"
        analysis_message += f"Average Energy: {bottleneck_energy:.3f} mJ per chiplet\n"
        analysis_message += f"Evidence: Observed in {len(supporting_kernels)} kernel(s): {', '.join(supporting_kernels)}\n\n"
        
        # Show top high energy consumers from the first kernel
        if energy_analysis:
            first_kernel = energy_analysis[0]
            analysis_message += f"Top High Energy Consumers ({first_kernel['kernel_name']} kernel):\n"
            for i, (chiplet_id, chiplet_data) in enumerate(first_kernel['top_consumers'], 1):
                chiplet_energy = chiplet_data.get('energy', 0)
                analysis_message += f"{i}. Chiplet {chiplet_id} ({chiplet_data.get('name', 'unknown')}): {chiplet_energy:.3f} mJ\n"
            analysis_message += "\n"
        
        # Show ONLY high energy consumers (above average threshold or top 3)
        sorted_types = sorted(all_type_stats.items(), key=lambda x: x[1]['avg_energy'], reverse=True)
        high_energy_types = []
        for chiplet_type, stats in sorted_types:
            if len(high_energy_types) < 3 or stats['avg_energy'] >= avg_threshold:
                high_energy_types.append((chiplet_type, stats))
        
        analysis_message += "High Energy Consumers (Causing Higher Energy Consumption):\n"
        for chiplet_type, stats in high_energy_types:
            analysis_message += f"- {chiplet_type.upper()}: {stats['avg_energy']:.3f} mJ avg ({stats['count']} chiplets)\n"
        analysis_message += "\nNote: Focus on these high energy consumers only. Low energy consumers are not shown.\n"
        
        self.messages.append({
            "role": "assistant",
            "content": analysis_message
        })

    def add_enhanced_runtime_analysis(self):
        """
        Add comprehensive runtime analysis context to help with runtime bottleneck questions.
        """
        if not self.pointInActiveContext or len(self.full_data) == 0:
            return
        
        # Analyze runtime distribution across all chiplets
        runtime_analysis = []
        for kernel_idx, kernel_data in enumerate(self.full_data):
            kernel_name = kernel_data.get('name', f'Kernel {kernel_idx}')
            chiplets = kernel_data.get('chiplets', {})
            
            # Sort chiplets by execution time
            sorted_chiplets = sorted(
                chiplets.items(), 
                key=lambda x: x[1].get('exe_time', 0), 
                reverse=True
            )
            
            # Get top runtime consumers
            top_consumers = sorted_chiplets[:3]  # Focus on top 3
            
            # Calculate runtime statistics
            all_runtimes = [chiplet[1].get('exe_time', 0) for chiplet in sorted_chiplets]
            total_runtime = sum(all_runtimes)
            
            # Group by chiplet type
            type_runtime = {}
            for chiplet_id, chiplet_data in sorted_chiplets:
                chiplet_type = chiplet_data.get('name', 'unknown')
                if chiplet_type not in type_runtime:
                    type_runtime[chiplet_type] = []
                type_runtime[chiplet_type].append(chiplet_data.get('exe_time', 0))
            
            # Calculate type-specific statistics
            type_stats = {}
            for chiplet_type, runtimes in type_runtime.items():
                type_stats[chiplet_type] = {
                    'count': len(runtimes),
                    'total_runtime': sum(runtimes),
                    'avg_runtime': sum(runtimes) / len(runtimes),
                    'percentage': (sum(runtimes) / total_runtime) * 100
                }
            
            runtime_analysis.append({
                'kernel_name': kernel_name,
                'kernel_number': kernel_data.get('kernal_number', kernel_idx),
                'total_runtime': total_runtime,
                'top_consumers': top_consumers,
                'type_statistics': type_stats
            })
        
        # Create concise runtime analysis message
        analysis_message = "Runtime Bottleneck Analysis:\n\n"
        
        # Identify the main bottleneck across all kernels
        all_type_stats = {}
        for analysis in runtime_analysis:
            for chiplet_type, stats in analysis['type_statistics'].items():
                if chiplet_type not in all_type_stats:
                    all_type_stats[chiplet_type] = {
                        'total_runtime': 0,
                        'count': 0,
                        'avg_runtime': 0
                    }
                all_type_stats[chiplet_type]['total_runtime'] += stats['total_runtime']
                all_type_stats[chiplet_type]['count'] += stats['count']
        
        # Calculate overall averages
        for chiplet_type in all_type_stats:
            all_type_stats[chiplet_type]['avg_runtime'] = (
                all_type_stats[chiplet_type]['total_runtime'] / 
                all_type_stats[chiplet_type]['count']
            )
        
        # Find the main bottleneck
        main_bottleneck = max(all_type_stats.items(), key=lambda x: x[1]['avg_runtime'])
        bottleneck_type = main_bottleneck[0]
        bottleneck_runtime = main_bottleneck[1]['avg_runtime']
        
        # Get supporting evidence from kernels
        supporting_kernels = []
        for analysis in runtime_analysis:
            if analysis['type_statistics'].get(bottleneck_type, {}).get('avg_runtime', 0) > 0:
                supporting_kernels.append(analysis['kernel_name'])
        
        analysis_message += f"Main Runtime Bottleneck: {bottleneck_type.upper()} chiplets\n"
        analysis_message += f"Average Runtime: {bottleneck_runtime:.3f} ms per chiplet\n"
        analysis_message += f"Evidence: Observed in {len(supporting_kernels)} kernel(s): {', '.join(supporting_kernels)}\n\n"
        
        # Show top 3 runtime consumers from the first kernel for quick reference
        if runtime_analysis:
            first_kernel = runtime_analysis[0]
            analysis_message += f"Top Runtime Consumers ({first_kernel['kernel_name']} kernel):\n"
            for i, (chiplet_id, chiplet_data) in enumerate(first_kernel['top_consumers'], 1):
                analysis_message += f"{i}. Chiplet {chiplet_id} ({chiplet_data.get('name', 'unknown')}): {chiplet_data.get('exe_time', 0):.3f} ms\n"
            analysis_message += "\n"
        
        # Quick runtime breakdown by type
        analysis_message += "Runtime Breakdown by Type:\n"
        for chiplet_type, stats in sorted(all_type_stats.items(), key=lambda x: x[1]['avg_runtime'], reverse=True):
            analysis_message += f"- {chiplet_type.upper()}: {stats['avg_runtime']:.3f} ms avg ({stats['count']} chiplets)\n"
        
        self.messages.append({
            "role": "assistant",
            "content": analysis_message
        })