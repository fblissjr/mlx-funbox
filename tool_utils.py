import json


def load_tools_from_file(file_path):
    with open(file_path, "r") as f:
        tools = json.load(f)
    return tools
