import base64
import os
from time import sleep
from uuid import uuid4

from langfuse.api import LangfuseAPI


def create_uuid():
    return str(uuid4())


def get_api():
    sleep(2)

    return LangfuseAPI(
        username=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        password=os.environ.get("LANGFUSE_SECRET_KEY"),
        base_url=os.environ.get("LANGFUSE_BASE_URL"),
    )


def encode_file_to_base64(image_path) -> str:
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")
