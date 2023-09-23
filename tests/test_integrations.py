from dotenv import load_dotenv
from langfuse.integrations import openai

load_dotenv()


def test_openai_chat_completion():
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": "1 + 1 = "}], temperature=0
    )
    assert len(completion.choices) != 0


def test_openai_completion():
    completion = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt="1 + 1 = ", temperature=0)
    assert len(completion.choices) != 0
