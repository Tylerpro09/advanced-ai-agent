from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from app.config import settings
from app.core.continual_learning import AdapterRegistry, LoRAContinualTrainer
from app.core.experience import ExperienceStore
from app.core.experience_policy import ExperiencePolicyNetwork
from app.core.human_learning import HumanLearningSystem
from app.core.memory import MemoryStore
from app.core.neural_memory import NeuralMemoryStore


def _safe_user(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", value)[:100] or "user"


async def run(args: argparse.Namespace) -> dict:
    memory = MemoryStore(settings.database_path)
    neural = NeuralMemoryStore(memory)
    experiences = ExperienceStore(memory, neural.embedder)
    policy = ExperiencePolicyNetwork(experiences)
    human = HumanLearningSystem(experiences, neural.embedder)

    rated = [x for x in experiences.list(args.user_id, args.limit) if abs(float(x.reward)) >= 0.15]
    positive = [x for x in rated if float(x.reward) >= float(settings.lora_min_reward)]

    result: dict = {
        "user_id": args.user_id,
        "rated_experiences": len(rated),
        "positive_lora_candidates": len(positive),
    }

    # Refit the small experience-policy network against all currently rated examples.
    result["experience_policy"] = policy.train_user(args.user_id)

    # Re-consolidate rated episodes into concepts/procedures/avoidances. The cognitive
    # store merges similar memories, so this reinforces useful patterns instead of
    # intentionally creating a second copy for every run.
    result["human_learning"] = await human.consolidate_existing(args.user_id, limit=args.limit)

    export_dir = settings.project_root / "data" / "experience_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"{_safe_user(args.user_id)}.jsonl"
    result["exported_examples"] = experiences.export_training_jsonl(
        args.user_id, str(export_path), args.min_reward
    )
    result["dataset"] = str(export_path)

    if args.lora:
        registry = AdapterRegistry()
        trainer = LoRAContinualTrainer(experiences, registry)
        try:
            result["lora"] = trainer.train(args.user_id, activate=args.activate)
        except Exception as exc:
            result["lora"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "hint": (
                    "Real LoRA weight training requires CONTINUAL_LEARNING_ENABLED=true, "
                    "LORA_BASE_MODEL=<original Hugging Face model>, and requirements-training.txt. "
                    "A GGUF loaded by LM Studio is an inference artifact and is not trained directly here."
                ),
            }

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strengthen accumulated learning and optionally train a versioned LoRA adapter."
    )
    parser.add_argument("--user-id", required=True, help="Agent user id whose experiences should be learned")
    parser.add_argument("--limit", type=int, default=2000, help="Maximum recent experiences to inspect")
    parser.add_argument("--min-reward", type=float, default=0.35, help="Minimum reward for exported training data")
    parser.add_argument("--lora", action="store_true", help="Also run real LoRA weight training")
    parser.add_argument("--no-activate", dest="activate", action="store_false", help="Do not activate a successful LoRA version")
    parser.set_defaults(activate=True)
    args = parser.parse_args()
    args.limit = max(1, min(100000, int(args.limit)))
    args.min_reward = max(-1.0, min(1.0, float(args.min_reward)))

    result = asyncio.run(run(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    lora = result.get("lora")
    return 1 if isinstance(lora, dict) and lora.get("ok") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
