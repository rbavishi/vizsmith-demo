import os


PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))
CACHE_DIR = f"{PROJECT_DIR}/.automl_cache"

os.makedirs(CACHE_DIR, exist_ok=True)
