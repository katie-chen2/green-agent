from wsgiref import types
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


class Agent:
    def __init__(self):
        # self.messenger = Messenger()
        # Initialize other state here
        load_dotenv()
        self.messages = []
        self.client = genai.Client()
        self.config = genai_types.GenerateContentConfig(
            system_instruction="You are an evaluator. Your task is to provide accurate and reliable evaluations for agentic responses to benchmark queries.",
        )

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)
        print(f"> Received message: {input_text}")
        self.messages.append(input_text)

        # Replace this example code with your agent logic

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )
        response = self.client.models.generate_content(
            model="gemini-3-flash-preview", contents=self.messages, config=self.config
        )
        print(response.text)
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=response.text if response.text else "No response"))],
            name="Response from Gemini-3",
        )
