import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

class ChatBotModel():
    def __init__(self, specs=None, model: str="gpt-4o-mini"):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.set_specs(specs=specs)
        self.messages = self._specs.copy()
        self._model = model

    def set_specs(self, specs):
        self._specs = [
            {
                "role": "developer",
                "content": "You are an expert in chiplet design and optimization. " + 
                           "Designs are made up of several chiplets, and the different numbers of different kinds of chiplets " +
                           "give different performance charactaristics. Designs are primarily evaluated based on " +
                           "performance and power consumption for a given trace. You are to help the user design a chiplet based on " +
                           "the information you are provided."
            }
        ] if specs is None else specs

    def set_model(self, model: str):
        self._model = model

    def get_response(self, content, role="user"):
        # messages = self._specs.copy()
        self.messages.append(
            {
                "role": role,
                "content": content
            }
        )
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
    
    def add_information(self, content, role="developer"):
        self.messages.append(
            {
                "role": role,
                "content": content
            }
        )
    
    def clear_history(self):
        self.messages = []
        self.messages = self._specs.copy()