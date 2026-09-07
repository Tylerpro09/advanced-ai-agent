from __future__ import annotations

import argparse
import json

from app.config import settings
from app.core.memory import MemoryStore
from app.core.neural_memory import NeuralMemoryStore
from app.core.experience import ExperienceStore
from app.core.continual_learning import AdapterRegistry, LoRAContinualTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a versioned LoRA adapter from positively-rated experiences.")
    parser.add_argument("--user", required=True, help="User ID whose experiences will be used")
    parser.add_argument("--no-activate", action="store_true", help="Train but do not mark the new adapter active")
    args = parser.parse_args()

    memory = MemoryStore(settings.database_path)
    neural = NeuralMemoryStore(memory)
    experiences = ExperienceStore(memory, neural.embedder)
    trainer = LoRAContinualTrainer(experiences, AdapterRegistry())
    result = trainer.train(args.user, activate=not args.no_activate)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
