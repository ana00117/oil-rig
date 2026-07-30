from collections import deque
from typing import Dict, List


class ConversationMemory:

    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages = deque(maxlen=max_messages)

    def add_user_message(self, message: str):
        self.messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

    def add_assistant_message(self, message: str):
        self.messages.append(
            {
                "role": "assistant",
                "content": message,
            }
        )

    def get_messages(self) -> List[Dict[str, str]]:
        return list(self.messages)

    def get_context(self) -> str:
        conversation = []

        for message in self.messages:
            role = (
                "User"
                if message["role"] == "user"
                else "Assistant"
            )

            conversation.append(
                f"{role}: {message['content']}"
            )

        return "\n".join(conversation)

    def clear(self):
        self.messages.clear()

    def __len__(self):
        return len(self.messages)
