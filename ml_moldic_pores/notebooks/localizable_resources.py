import os

from dotenv import load_dotenv
load_dotenv(".env", override=True)
lang_code: str = os.getenv("LOCALIZATION_LANGUAGE") or "en-US"

class str:
    pass