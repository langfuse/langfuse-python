import os
from uuid import uuid4

from langfuse.api.client import FernLangfuse


def create_uuid():
    return str(uuid4())


def get_api():
    return FernLangfuse(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_HOST"),
    )
