import os
from uuid import uuid4

from langfuse.api.client import FintoLangfuse


def create_uuid():
    return str(uuid4())


def get_api():
    print(os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY"), os.environ.get("LANGFUSE_HOST"))
    return FintoLangfuse(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        environment=os.environ.get("LANGFUSE_HOST"),
    )
