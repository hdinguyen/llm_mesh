from dspy import Module, LM, ChainOfThought
from translate_signature import TranslateSignature
import os

from collections.abc import AsyncGenerator

from acp_sdk.models import Metadata, Annotations, Message, MessagePart
from acp_sdk.models.platform import PlatformUIAnnotation, PlatformUIType
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

class TranslateModule(Module):
    def __init__(self):
        super().__init__()
        llm = LM(
            model="openrouter/deepseek/deepseek-chat-v3-0324:free",
            api_key=os.getenv("KEY"),
        )
        self.cot = ChainOfThought(TranslateSignature)
        self.cot.set_lm(llm)

    def forward(self, input: str, translate_goal: str) -> str:
        return self.cot(input=input, translate_goal=translate_goal)
    
server = Server()

@server.agent(
    name="translate_agent",
    description=(
        "Translate input text from raw language to target language that maintain the same length, meaning and context"
    ),
    input_content_types=["text/plain", "text/plain"],
    output_content_types=["text/plain"],
    metadata=Metadata(
        annotations=Annotations(
            beeai_ui=PlatformUIAnnotation(
                ui_type=PlatformUIType.CHAT,
            )
        ),
        programming_language="python",
        framework="dspy",
        documentation="TBD",
    ),
)
async def translate_agent(input: list[Message], context: Context) -> AsyncGenerator[RunYield, RunYieldResume]:
    input_text = input[0].parts[0].content

    try:
        translate_target = input[0].parts[1].content
        raw_input = input[0].parts[0].content
    except IndexError:
        translate_target = input[0].parts[0].content
        raw_input = input[0].parts[1].content
    except Exception as e:
        yield Message(
            role="agent/translation",
            parts=[
                MessagePart(
                    role="agent/translation",
                    content=f"Error: {e}",
                    content_type="text/plain",
                )
            ],
        )

    try:
        translated_text = TranslateModule().forward(input=raw_input, translate_goal=translate_target)
        yield Message(
            role="agent/translation",
            parts=[
                MessagePart(
                    role="agent/translation",
                    content=translated_text.output,
                    content_type="text/plain",
                )
            ],
        )
    except Exception as e:
        yield Message(
            role="agent/translation",
            parts=[
                MessagePart(
                    role="agent/translation",
                    content=f"Error: {e}",
                    content_type="text/plain",
                )
            ],
        )


server.run()