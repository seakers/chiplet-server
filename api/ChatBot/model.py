import os
import dotenv
from openai import OpenAI
import json

dotenv.load_dotenv()

class ChatBotModel():

    def __init__(self, specs=None, model: str="gpt-4o-mini"):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._adj_input = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.set_specs(specs=specs)
        self.adj_input_specs()
        self.messages = self._specs.copy()
        self.adj_input_messages = self._adj_input.copy()
        self._model = model
        self.pointContext = {}
        self.pointInActiveContext = False

    def adj_input_specs(self):
        self._adj_input_specs = [
            {
                "role": "developer",
                "content": "You are an expert in chiplet design and optimization. " +
                           "Users will come to you with vague questions about chiplet design, and your only job is to rewrite thier questions to be more specific, " +
                           "so that the questions can be closely matched to an existing dataset. One example line of the dataset is: " +
                           "\n {'name': 'Attention', 'chiplets': {0: {'flops': np.float64(7305919517376.859), 'mem_accessed': np.float64(581855534.4761195), 'exe_time': np.float64(0.24353065057922865), 'energy': np.float64(2.631951106194661), 'work': np.float64(0.15820702601508768), 'name': 'atten'}, 1: {'flops': np.float64(7305919517376.859), 'mem_accessed': np.float64(581855534.4761195), 'exe_time': np.float64(0.24353065057922865), 'energy': np.float64(2.631951106194661), 'work': np.float64(0.15820702601508768), 'name': 'atten'}, 2: {'flops': np.float64(7305919517376.859), 'mem_accessed': np.float64(581855534.4761195), 'exe_time': np.float64(0.24353065057922865), 'energy': np.float64(2.631951106194661), 'work': np.float64(0.15820702601508768), 'name': 'atten'}, 3: {'flops': np.float64(7305919517376.859), 'mem_accessed': np.float64(581855534.4761195), 'exe_time': np.float64(0.24353065057922865), 'energy': np.float64(2.631951106194661), 'work': np.float64(0.15820702601508768), 'name': 'atten'}, 4: {'flops': np.float64(7305919517376.859), 'mem_accessed': np.float64(581855534.4761195), 'exe_time': np.float64(0.24353065057922865), 'energy': np.float64(2.631951106194661), 'work': np.float64(0.15820702601508768), 'name': 'atten'}, 5: {'flops': np.float64(244301057223.21262), 'mem_accessed': np.float64(553942701.0910256), 'exe_time': np.float64(0.2443010572232126), 'energy': np.float64(0.6967985892026394), 'work': np.float64(0.005290250409096115), 'name': 'sparse'}, 6: {'flops': np.float64(244301057223.21262), 'mem_accessed': np.float64(553942701.0910256), 'exe_time': np.float64(0.2443010572232126), 'energy': np.float64(0.6967985892026394), 'work': np.float64(0.005290250409096115), 'name': 'sparse'}, 7: {'flops': np.float64(1832257733052.2563), 'mem_accessed': np.float64(560219501.5674703), 'exe_time': np.float64(0.24430103107363418), 'energy': np.float64(1.7582047001505203), 'work': np.float64(0.03967687382127389), 'name': 'conv'}, 8: {'flops': np.float64(1832257733052.2563), 'mem_accessed': np.float64(560219501.5674703), 'exe_time': np.float64(0.24430103107363418), 'energy': np.float64(1.7582047001505203), 'work': np.float64(0.03967687382127389), 'name': 'conv'}, 9: {'flops': np.float64(1832257733052.2563), 'mem_accessed': np.float64(560219501.5674703), 'exe_time': np.float64(0.24430103107363418), 'energy': np.float64(1.7582047001505203), 'work': np.float64(0.03967687382127389), 'name': 'conv'}, 10: {'flops': np.float64(1832257733052.2563), 'mem_accessed': np.float64(560219501.5674703), 'exe_time': np.float64(0.24430103107363418), 'energy': np.float64(1.7582047001505203), 'work': np.float64(0.03967687382127389), 'name': 'conv'}, 11: {'flops': np.float64(1832257733052.2563), 'mem_accessed': np.float64(560219501.5674703), 'exe_time': np.float64(0.24430103107363418), 'energy': np.float64(1.7582047001505203), 'work': np.float64(0.03967687382127389), 'name': 'conv'}}, 'total': {'flops': np.float64(46179488366592.01), 'mem_accessed': np.float64(735513149.44), 'exe_time': np.float64(0.2443010572232126), 'energy': np.float64(23.34437621013119), 'energy_dram': np.float64(1.9808869311655408)}}" +
                           "\n where 'flops' is floating-point operations per second, 'mem_accessed' is the amount of memory accessed, " +
                           "'exe_time' is the execution time, 'energy' is the energy used, and 'energy_dram' relates to the dynamic random access memory. " +
                           "For example, if the user asks 'what is the main bottleneck of this design in terms of execution time?', you might rewrite it to: " + 
                           "Where in this design is 'flops' low and 'exe_time' high? " +
                           "When rewriting the user's question, make it as specific as possible to match existing dataset entries. " +
                           "Keep your responses as consise as possible while still providing all the relevant information, " +
                           "as if you are talking to another expert in the field."
            }
        ]

    def set_specs(self, specs):
        self._specs = [
            {
                "role": "developer",
                "content": "You are an expert in chiplet design and optimization. " +
                           "Designs are made up of several chiplets, and the different numbers of different kinds of chiplets " +
                           "give different performance charactaristics. Designs are primarily evaluated based on " +
                           "performance and power consumption for a given trace. You are to help the user design a chiplet based on " +
                           "the information you are provided. Keep your responses as consise as possible while still providing all the relevant information, " +
                           "as if you are talking to another expert in the field. "
            }
        ] if specs is None else specs

    def set_model(self, model: str):
        self._model = model

    def get_response(self, content, role="user"):
        if self.pointInActiveContext:
            # Check if the content is similar to any point in the context
            relevant_context = ''
            # content_embedding = self.get_embedding(content)
            rewritten_content = self._adj_input.chat.completions.create(
                model=self._model,
                messages=self._adj_input_specs + [{"role": "user", "content": content}]
            )
            rewritten_message = rewritten_content.choices[0].message.content
            content_embedding = self.get_embedding(rewritten_message)
            print(f"Rewritten message: {rewritten_message}")
            print(f"Length of pointContext: {len(self.pointContext)}")
            rel_context = {}
            for point_str in self.pointContext:
                similarity = self.cosine_similarity(content_embedding, self.pointContext[point_str])
                # print(f"Similarity: {similarity}")
                if len(rel_context) < 5:
                    rel_context[point_str] = similarity
                elif similarity > min(rel_context.values()):
                    min_key = min(rel_context, key=rel_context.get)
                    del rel_context[min_key]
                    rel_context[point_str] = similarity
                # if similarity > 0.8:
                #     relevant_context = relevant_context + point_str + '\n'
            print(f"Length of rel_context: {len(rel_context)}")
            for point_str in rel_context:
                relevant_context += point_str + '\n'
        if self.pointInActiveContext:
            self.messages.append(
                {
                    "role": role,
                    "content": rewritten_message
                }
            )
        else:
            self.messages.append(
                {
                    "role": role,
                    "content": content
                }
            )
        if self.pointInActiveContext and relevant_context:
            print("Relevant context found.")
            self.messages.append(
                {
                    "role": "user",
                    "content": "Here are some of the most relevant kernals. Here 'name' gives the kernal type, 'flops' is floating-point operations per second, " +
                               "'mem_accessed' is the amount of memory accessed, 'exe_time' is the execution time, 'energy' is the energy used, " +
                               "and 'energy_dram' relates to the dynamic random access memory. When using this context, " +
                               "make your responses specific to the kernal and chiplet, and be as consise as possible while still providing all the relevant information." +
                               f"\n\n {str(relevant_context)}"
                }
            )
        else:
            print("No relevant context found.")
        # print(self.messages)
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
        self.pointContext = {} # reset to zero in case of multiple calls
        self.pointInActiveContext = True
        try:
            with open(contextFilePath, 'r') as file:
                batchSize = 90 # maximum size for this embedding model is 8192, batch size of 90 puts it just under that
                data = json.load(file)
                for i in range(0, len(data), batchSize):
                    if i + batchSize > len(data):
                        batch = data[i:]
                    else:
                        batch = data[i:i+batchSize]
                    for dict in batch:
                        if 'chiplets' in dict:
                            del dict['chiplets']
                    # if i == 0:
                    #     print(f"Batch: {batch}")
                    batch_str = json.dumps(batch)
                    embedding = self.get_embedding(batch_str)
                    self.pointContext[batch_str] = embedding
                    print("Got embedding for an entry")
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
        except FileNotFoundError:
            print(f"Error: File at {contextFilePath} not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

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