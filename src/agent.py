import json
from wsgiref import types
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message
import asyncio
from messenger import Messenger


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here
        load_dotenv()
        self.messages = []
        self._by_sender: dict[str, list[str]] = {}
        self.client = genai.Client()
        self.kernel_config = genai_types.GenerateContentConfig(
            system_instruction="You are a monitor. Your task is to handle the request of each purple agent and maintain the memory context." \
            "Output an updated memory context based on your understanding of the requests and contexts.",
        )
        self.evaluator_config = genai_types.GenerateContentConfig(
            system_instruction="You are an evaluator. Your task is to attribute each agentic logic block in the memory context to the correct purple agent." \
            "Provide accurate, reliable attributions based on the context.",
        )

    async def _poll_contenders(self, urls: list[str], message: str) -> list[Message]:
        """Poll contender agents and return their messages."""
        responses = await asyncio.gather(*[
            self.messenger.talk_to_agent(message, url) for url in urls
        ])
        return responses

    async def _handle_incoming(self, message: Message):
        """Build sender-prefixed combined text and accumulate per-sender history."""
        ordered_concat_parts: list[str] = []

        default_sender = (message.metadata or {}).get("sender") if message.metadata else None
        default_type = (message.metadata or {}).get("type") if message.metadata else None

        for _, part in enumerate(message.parts):
            root = part.root
            part_meta = getattr(root, "metadata", None)
            sender = default_sender
            msg_type = default_type

            if isinstance(part_meta, dict):
                sender = part_meta.get("sender", sender)
                msg_type = part_meta.get("type", msg_type)

            sender_key = sender or "unknown"
            text = root.text if isinstance(root, TextPart) else None

            if text:
                ordered_concat_parts.append(f"{sender_key}: {text}")
                self._by_sender.setdefault(sender_key, []).append(text)

        # Snapshot current grouping into message metadata for downstream consumers
        meta = dict(message.metadata or {})
        meta["grouped_by_sender"] = {k: list(v) for k, v in self._by_sender.items()}
        message.metadata = meta

        combined = "\n".join(ordered_concat_parts) if ordered_concat_parts else get_message_text(message)
        return {"combined": combined, "grouped_by_sender": meta["grouped_by_sender"]}

    def _parse_attribution_json(self, text: str) -> dict | None:
        """Parse evaluator JSON output; fallback to best-effort extraction."""
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            pass

        # Best-effort: extract first JSON object from text
        import re

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
        return None

    async def _handle_memory_kernel(self):
        """Summarize message history."""
        if len(self.messages) < 2:
            return  # Not enough to summarize
        
        # Create updated memory context
        history_text = "\n---\n".join(self.messages)
        summary_prompt = "Here is the conversation history:\n" + history_text + \
            "\n\nPlease provide an updated memory context of the requests and contexts."
        
        summary_response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[summary_prompt],
            config=self.kernel_config
        )
        
        # Replace all old messages with updated memory context
        self.messages = [
            f"[SUMMARY]\n{summary_response.text}"
        ]

    async def _evaluate_memory_attributions(self, by_sender: dict[str, list[str]]):
        """Evaluate attributions and return (attributions_dict, game_won, raw_text)."""
        if not self.messages:
            return None, False, ""  # No messages to evaluate

        memory_text = "\n---\n".join(self.messages)
        sender_contributions = ""
        for sender, texts in by_sender.items():
            sender_contributions += f"\nSender: {sender}\nContributions:\n" + "\n".join(texts) + "\n"
        evaluation_prompt = (
            "Here is the current memory context:\n" + memory_text
            + "\n\nPlease attribute each agentic logic block to the correct purple agent."
            + "\n\nHere are the contributions of each agent:" + sender_contributions
            + "\n\nOutput the percentage attribution for each agent in json format."
            + "\nExample output:\n{\n  'agent_1': 60,\n  'agent_2': 40\n}"
        )
        evaluation_response = self.client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[evaluation_prompt],
            config=self.evaluator_config
        )

        raw_text = evaluation_response.text or ""
        attributions = self._parse_attribution_json(raw_text)

        game_won = False
        if isinstance(attributions, dict) and attributions:
            try:
                best = max(float(v) for v in attributions.values())
                game_won = best > 80.0
            except Exception:
                game_won = False

        print(f"Attribution Evaluation:\n{raw_text}")
        print(f"Parsed attributions: {attributions}, game_won={game_won}")
        return attributions, game_won, raw_text


    async def run(self, message: Message, updater: TaskUpdater) :
        """
        The green agent is a monitor agent that evaluates the context to 
         determine which logic block comes from which purple agent.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        # Check if message has a sender in metadata
        sender = (message.metadata or {}).get("sender") if message.metadata else None
        
        if not sender:
            # No sender - poll other agents with this message
            text = get_message_text(message)
            agent_urls = ['http://localhost:8008']  # Configure these URLs as needed
            print(f"> Polling agents for message without sender: {text}")
            responses = await self._poll_contenders(agent_urls, text)
            print(f"> Received {len(responses)} responses from agents")
            # Intentionally do not add to memory or emit attribution
            return
        
        # Has sender - proceed with memory management and attribution
        extracted = await self._handle_incoming(message)
        combined = extracted["combined"]
        by_sender = extracted["grouped_by_sender"]

        print(f"> Received message: {combined}")
        self.messages.append(combined)

        # Update memory context and evaluate attributions
        await self._handle_memory_kernel()
        attributions, game_won, raw_eval = await self._evaluate_memory_attributions(by_sender)
        if attributions is not None:
            print(f"Game won: {game_won}")
            # Emit a status update with an agent message containing the attribution summary
            summary_text = json.dumps({
                "game_won": game_won,
                "attributions": attributions,
            })
            summary_msg = updater.new_agent_message(parts=[Part(TextPart(text=summary_text))])
            await updater.update_status(TaskState.working, message=summary_msg)
            # Optionally emit raw evaluation details for inspection
            if raw_eval:
                eval_msg = updater.new_agent_message(parts=[Part(TextPart(text=f"[EVALUATION]\n{raw_eval}"))])
                await updater.update_status(TaskState.working, message=eval_msg)
        # No explicit return; results are communicated via TaskUpdater events