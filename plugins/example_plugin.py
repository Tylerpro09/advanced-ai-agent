TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "project_status",
        "description": "Example plugin that demonstrates the plugin API.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def run(args, context):
    return f"Plugin system is active for user {context.get('user_id', 'unknown')}."
