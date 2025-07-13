from dspy import Signature, InputField, OutputField

class TranslateSignature(Signature):
    input = InputField(
        description="The input text to translate",
    )
    translate_goal = InputField(
        description="This is the destination translation language of the input text",
    )
    output = OutputField(
        description="The translated text",
    )
