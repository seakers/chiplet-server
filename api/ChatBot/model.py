import os
import dotenv
from openai import OpenAI

dotenv.load_dotenv()

class ChatBotModel():
    def __init__(self, specs=None, model: str="gpt-4o-mini"):
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.set_specs(specs=specs)
        self._model = model

    def set_specs(self, specs):
        self._specs = [
            {
                "role": "developer",
                "content": "You are a helpful assistant. You respond precisely to user queries, with just a few words."
            }
        ] if specs is None else specs

    def set_model(self, model: str):
        self._model = model

    def get_response(self, content, role="user"):
        messages = self._specs.copy()
        messages.append(
            {
                "role": role,
                "content": content
            }
        )
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages
        )

        return completion.choices[0].message.content