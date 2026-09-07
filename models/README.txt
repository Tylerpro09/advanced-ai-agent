Place your GGUF chat/instruct model here.

Default expected file:
    models/model.gguf

Or set an absolute/relative path in .env:
    LOCAL_MODEL_PATH=models/your-model.gguf

The GGUF file contains the neural-network weights. The agent itself cannot be a capable LLM without model weights somewhere; embedded mode removes the need for LM Studio/Ollama/a remote API.
