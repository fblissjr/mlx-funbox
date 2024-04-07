import json


def load_tools_from_file(file_path):
    with open(file_path, "r") as f:
        tools = json.load(f)
    return tools


def format_tools(tools):
    formatted_tools = "\n".join(
        [
            f"```python\ndef {tool['name']}("
            + ", ".join(
                [
                    f"{param}: {param_info['type']}"
                    for param, param_info in tool["parameter_definitions"].items()
                ]
            )
            + f") -> List[Dict]:\n    \"\"\"{tool['description']}\"\"\"\n    pass\n```"
            for tool in tools
        ]
    )
    return f"""
## Available Tools
Here is a list of tools that you have available to you:

{formatted_tools}
"""
