# Security

Do not commit `.env`, API tokens, bot tokens, private documents, `data/`, model weights, or personal LoRA adapters.

The HTTP tool is deny-by-default and requires `HTTP_ALLOWLIST`. Private/loopback/link-local/reserved network targets are blocked unless `HTTP_ALLOW_PRIVATE_NETWORKS=true` is explicitly set.

PC execution tools are disabled by default. If enabled, `PC_COMMAND_ALLOWLIST` controls which executables may run and commands are launched with `shell=False`.

If the API is reachable from another device, configure `API_TOKEN` and prefer HTTPS through a trusted reverse proxy or private network overlay.
