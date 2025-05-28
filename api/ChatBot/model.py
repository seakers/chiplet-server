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
    

    def rule_mining(self, point_selection=None):
        """
        find combinations of features which have high supp and are on the pareto front of
        conf(f->p) and conf(p->f)
        where f are combinations of features and p is a selected set of values
        p will usually be the first few ranks of the pareto front but can also be user selected points
        features are, for each chiplet type
        num = 0 (none)
        1 <= num <= 2 (low)
        3 <= num <= 5 (medium)
        6 <= num <= 8 (high)
        num >= 9 (very high)
        """
        file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            full_data = []
            for row in csv_reader:         
                full_data.append(
                    (float(row[0]), float(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]))  # exe_time, energy, GPU, Attention, Sparse, Convolution
                )
        full_data = np.array(list(set(full_data)))
        
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

        if point_selection is None: # get the first three ranks of the pareto front
            point_selection = full_data[full_data[:, -1] < 3]  # select the first three ranks of the pareto front


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