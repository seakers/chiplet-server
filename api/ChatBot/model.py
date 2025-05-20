import os
import dotenv
from openai import OpenAI
import json
import csv
import numpy as np

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
                           "and the number of values you want to return. Do not respond with anythin except the data request."
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

    def get_pareto_front_questions(self):
        pfront_question_list = [
            "Is this design Pareto optimal?",
            "What is the rank of this solution?",
            "How good is this point?",
            "Where are the best designs?",
            "Why is this design not optimal?",
            "What are some common features of the Pareto front?",
            "Based on the existing Pareto front, can you propose a new design?"
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
                           "give different performance charactaristics. Designs are primarily evaluated based on " +
                           "performance and power consumption for a given trace. You are to help the user design a chiplet based on " +
                           "the information you are provided. Keep your responses as concise as possible, " +
                           "as if you are talking to another expert in the field. "
            }
        ] if specs is None else specs

    def set_model(self, model: str):
        self._model = model

    def get_response(self, content, role="user"):
        self.messages.append(
            {
                "role": role,
                "content": content
            }
        )

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

        
        # Add pareto context if the content is similar to any of the pareto front questions
        content_embedding = self.get_embedding(content)
        for question in self.pareto_front_questions:
            pareto_similarity = self.cosine_similarity(content_embedding, self.pareto_front_questions[question])
            print(f"Question: {question}")
            print(f"Pareto similarity: {pareto_similarity}")
            if pareto_similarity > 0.7:
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
            point_vals = []
            for row in csv_reader:
                point_vals.append([float(row[0]), float(row[1])])
                full_data.append(
                    {
                        "exe_time": float(row[0]),
                        "energy": float(row[1]),
                        "chiplets": {
                            "GPU": int(row[2]),
                            "Attention": int(row[3]),
                            "Sparse": int(row[4]),
                            "Convolution": int(row[5])
                        }
                    }
                )

        pfront_data = []
        prank = 0
        point_vals = np.array(point_vals)
        print(f"Full data: {full_data}")
        # print(f"Point vals: {point_vals}")
        while len(point_vals) > 0:
            pfront_mask = self.is_pareto_efficient(point_vals)
            next_prank = point_vals[pfront_mask]
            pfront_data.extend(next_prank)
            point_vals = point_vals[~pfront_mask]
            for data in full_data:
                if [data["exe_time"], data["energy"]] in next_prank:
                    data["rank"] = prank
            prank += 1

        print(f"Pareto front data: {full_data}")
        self.messages.append(
            {
                "role": "user",
                "content": "Here is some information about each point including their performance charactaristics " +
                           "(exe_time, energy), thier pareto ranking (rank), and the number of each kind of chiplet in the design " + 
                           "(GPU, Attention, Sparse, Convolution). The number of each kind of chiplet are the design variables, " +
                           "but there are always twelve chiplets in total. " + 
                           "When answering questions, keep your responses as concise as possible and do not restate things I already know. \n" +
                           f"{str(full_data)}"
            }
        )

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