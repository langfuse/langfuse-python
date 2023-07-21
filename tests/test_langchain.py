import os
from langfuse import Langfuse
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langfuse.callback import CallbackHandler


# @pytest.mark.asyncio
def test_callback_simple_chain():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", "http://localhost:3000")

    handler = CallbackHandler(langfuse)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAPI_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    review = synopsis_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    print("output variable: ", review)


# @pytest.mark.asyncio
def test_callback_sequential_chain():
    langfuse = Langfuse("pk-lf-1234567890", "sk-lf-1234567890", "http://localhost:3000")

    handler = CallbackHandler(langfuse)

    llm = OpenAI(openai_api_key=os.environ.get("OPENAPI_KEY"))
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {title}
        Playwright: This is a synopsis for the above play:"""

    prompt_template = PromptTemplate(input_variables=["title"], template=template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

        Play Synopsis:
        {synopsis}
        Review from a New York Times play critic of the above play:"""
    prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
    review_chain = LLMChain(llm=llm, prompt=prompt_template)

    overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain])
    review = overall_chain.run("Tragedy at sunset on the beach", callbacks=[handler])

    print(review)

    print("output variable: ", review)
