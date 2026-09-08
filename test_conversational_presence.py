from app.providers.openai_compatible import OpenAICompatibleProvider


def main():
    provider = OpenAICompatibleProvider("http://127.0.0.1:1234/v1", "", "auto")

    conversation = [
        {"role": "system", "content": "You are an advanced, local-first AI assistant with persistent memory."},
        {"role": "user", "content": "Quiero entrenar mejor mi modelo con LM Studio"},
        {"role": "assistant", "content": "Podemos usar experiencias y luego LoRA."},
        {"role": "user", "content": "hola"},
    ]
    instruction = provider._presence_instruction(conversation)
    assert instruction
    assert "help-desk" in instruction
    assert "generic help question" in instruction

    canned = {"role": "assistant", "content": "¡Hola! 👋 ¿En qué puedo ayudarte hoy?"}
    fixed = provider._humanize_canned_greeting(conversation, canned)
    assert "En qué puedo ayudarte" not in fixed["content"]
    assert "lo de antes" in fixed["content"]

    fresh = [
        {"role": "system", "content": "You are an advanced, local-first AI assistant with persistent memory."},
        {"role": "user", "content": "hola"},
    ]
    fixed_fresh = provider._humanize_canned_greeting(fresh, canned)
    assert fixed_fresh["content"] == "¡Hey! 👋 ¿Qué tal?"

    internal = [
        {"role": "system", "content": "Reflect on a rated AI experience and return JSON."},
        {"role": "user", "content": "hola"},
    ]
    assert provider._presence_instruction(internal) is None
    assert provider._humanize_canned_greeting(internal, canned) == canned

    print("Conversational presence tests passed")


if __name__ == "__main__":
    main()
