"""Train one dev-only DPO adapter per model, prompt, and learning rate."""

import argparse
import gc
from hashlib import sha256
from itertools import count
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import paths
from common.data import is_answerable, load_queries
from hybrid_claims.llm_generate import DESIGN, partition, write_json
from hybrid_claims.split_training_generation import DEMO_LINES, demo_messages, user_turn


OUT = Path(paths.OUT) / "split_training"
MODELS = {
    "3b": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    "7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
LEARNING_RATES = (5e-6, 2e-5)
LORA_SETTINGS = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": "all-linear"}
CV_PRECISIONS = {"zero": "bf16", "few": "nf4"}


def evidence_articles(question):
    return frozenset(evidence["title"] for evidence in question["evidence_list"])


def verified_cv_fold(manifest, fold_name, queries):
    """Require an exhaustive outer split and a separate inner checkpoint-selection split."""
    demos = set(DEMO_LINES)
    if sorted(manifest["excluded_demo_qids"]) != sorted(demos):
        raise ValueError("CV must reserve the four fixed demonstrations")
    eligible = {row["qid"] for row in queries if is_answerable(row)} - demos
    if manifest["eligible_qids"] != sorted(eligible):
        raise ValueError("CV eligibility must cover every non-demonstration answerable question")
    fold = manifest["folds"][fold_name]
    train, heldout, test = (set(fold[part]) for part in ("train", "heldout", "test"))
    if (not all((train, heldout, test)) or train | heldout | test != eligible
            or sum(len(fold[part]) for part in ("train", "heldout", "test")) != len(eligible)):
        raise ValueError("CV train, validation and test must partition the eligible questions")

    def groups(ids):
        return {frozenset(evidence["url"] for evidence in queries[qid]["evidence_list"]) for qid in ids}

    train_groups, heldout_groups, test_groups = map(groups, (train, heldout, test))
    if train_groups & heldout_groups or train_groups & test_groups or heldout_groups & test_groups:
        raise ValueError("An identical evidence-article set appears in multiple CV partitions")
    if fold_name == "article_disjoint":
        train_articles = set().union(*train_groups, *heldout_groups)
        if train_articles & set().union(*test_groups):
            raise ValueError("A gold article appears on both sides of the article-disjoint split")
    else:
        other_name = {"fold_1": "fold_2", "fold_2": "fold_1"}[fold_name]
        if set(manifest["folds"][other_name]["test"]) != train | heldout:
            raise ValueError("The two outer test folds must be complementary")
    return fold


def verified_dev_train(split, queries):
    dev = set(partition(queries, "dev"))
    demos = set(DEMO_LINES)
    if not demos <= dev:
        raise ValueError("A few-shot example is outside the fixed dev split")
    demo_articles = {evidence_articles(queries[qid]) for qid in demos}
    eligible = {qid for qid in dev if evidence_articles(queries[qid]) not in demo_articles}
    train_ids, heldout_ids = split["train"], split["heldout"]
    train, heldout = set(train_ids), set(heldout_ids)
    if (train & heldout or (train | heldout) != eligible or len(train) != len(train_ids)
            or len(heldout) != len(heldout_ids)):
        raise ValueError("Train and held-out must partition the eligible dev questions without duplicates")
    if ({evidence_articles(queries[qid]) for qid in train}
            & {evidence_articles(queries[qid]) for qid in heldout}):
        raise ValueError("An evidence-article group appears in both train and held-out")
    return train


def verified_pairs(bundle, split, variant, *, cv_fold=None):
    """Return conversational TRL rows after checking all IDs and prompt boundaries."""
    queries = load_queries()
    if cv_fold is None:
        train = verified_dev_train(split, queries)
    else:
        train = set(verified_cv_fold(split, cv_fold, queries)["train"])
        if (bundle.get("model") != "3b" or bundle.get("prompt") != variant
                or bundle.get("base_precision") != CV_PRECISIONS[variant]
                or bundle.get("samples_per_train_question") != 8):
            raise ValueError("CV preferences must use the matching 3B prompt and precision with eight draws")

    history = demo_messages() if variant == "few" else []
    seen = set()
    rows = []
    for pair in bundle["pairs"]:
        qid = pair["qid"]
        if type(qid) is not int or qid not in train or qid in seen:
            raise ValueError(f"Preference pair has an invalid or repeated training qid {qid}")
        seen.add(qid)
        prompt = pair["prompt"]
        if prompt != [*history, user_turn(queries[qid]["query"])]:
            raise ValueError(f"Preference prompt for qid {qid} does not match the {variant} protocol")

        chosen, rejected = pair["chosen"], pair["rejected"]
        if (not isinstance(chosen, str) or not isinstance(rejected, str) or chosen == rejected
                or pair["chosen_recall"] != 1 or not 0 <= pair["rejected_recall"] < 1):
            raise ValueError(f"Preference pair for qid {qid} fails the strict recall rule")
        rows.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    return rows


def load_base_model(model_name, precision):
    """Load the pinned Qwen base and tokenizer for adapter training or evaluation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 compute support is required for split training")
    model_id, revision = MODELS[model_name]
    cache = {"revision": revision, "local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **cache)
    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}, "attn_implementation": "sdpa"}
    if precision == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **cache, **kwargs)
    return model, tokenizer


def build_lora_config():
    from peft import LoraConfig

    return LoraConfig(task_type="CAUSAL_LM", **LORA_SETTINGS)


def checkpoint_records(attempt_dir, run_dir):
    checkpoints = []
    for checkpoint in attempt_dir.glob("checkpoint-*"):
        state_path = checkpoint / "trainer_state.json"
        adapter_path = checkpoint / "adapter_model.safetensors"
        if not state_path.is_file() or not adapter_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        checkpoints.append({"path": checkpoint.relative_to(run_dir).as_posix(),
                            "step": state["global_step"], "epoch": state["epoch"]})
    return sorted(checkpoints, key=lambda info: info["step"])


def dpo_training_args(config_type, attempt_dir, learning_rate, epochs):
    return config_type(
        output_dir=str(attempt_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        beta=0.1,
        learning_rate=learning_rate,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=1,
        disable_dropout=False,
        max_length=None,
        save_strategy="epoch",
        save_only_model=True,
        eval_strategy="no",
        report_to="none",
        push_to_hub=False,
        dataloader_num_workers=0,
        logging_steps=5,
        seed=DESIGN["seed"],
        data_seed=DESIGN["seed"],
    )


def memory_error(error):
    import torch

    if isinstance(error, torch.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and any(
        phrase in str(error).lower() for phrase in ("cuda out of memory", "cublas_status_alloc_failed")
    )


def preference_dataset(rows, pairs_sha256, directory=OUT / "preferences" / "trainer_data"):
    """Keep TRL's tokenization and reference log-prob cache across learning rates."""
    from datasets import Dataset, load_from_disk

    path = directory / pairs_sha256
    if not (path / "state.json").exists():
        Dataset.from_list(rows).save_to_disk(str(path))
    return load_from_disk(str(path))


def attempt_directory(run_dir, precision, epochs):
    for attempt in count():
        name = f"{precision}-restart-{attempt}" if attempt else precision
        folder = run_dir / name
        checkpoints = checkpoint_records(folder, run_dir)
        if len(checkpoints) == epochs and checkpoints[-1]["epoch"] >= epochs - 1e-6:
            return folder, checkpoints
        if not any(folder.glob("checkpoint-*")):
            return folder, []


def train_attempt(args, dataset, run_dir, precision):
    import torch
    from trl import DPOConfig, DPOTrainer

    attempt_dir, checkpoints = attempt_directory(run_dir, precision, args.epochs)
    attempt_dir.mkdir(exist_ok=True)
    if checkpoints:
        return {"status": "complete", "precision": precision, "checkpoints": checkpoints}
    if attempt_dir.name != precision:
        print(f"Restarting {precision} from the base model in {attempt_dir.name}", flush=True)
    started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        base, tokenizer = load_base_model(args.model, precision)
        trainer = DPOTrainer(
            model=base,
            ref_model=None,
            args=dpo_training_args(DPOConfig, attempt_dir, args.lr, args.epochs),
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=build_lora_config(),
        )
        trainer.train()
    except Exception as error:
        if not memory_error(error):
            raise
        result = {"status": "oom", "reason": str(error)}
    else:
        result = {"status": "complete"}

    checkpoints = checkpoint_records(attempt_dir, run_dir)
    if result["status"] == "complete" and (
        len(checkpoints) != args.epochs or checkpoints[-1]["epoch"] < args.epochs - 1e-6
    ):
        raise RuntimeError("DPO completed without saving the expected epoch checkpoints")
    result.update(precision=precision, checkpoints=checkpoints, seconds=perf_counter() - started,
                  peak_vram_gib=torch.cuda.max_memory_reserved() / (1024 ** 3))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--prompt", choices=("zero", "few"), required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lr", type=float, choices=LEARNING_RATES, required=True)
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--precision", choices=("auto", "bf16", "nf4"), default="auto")
    parser.add_argument("--cv-fold", choices=("fold_1", "fold_2", "article_disjoint"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    run_dir = args.output_dir.resolve()
    if not run_dir.is_relative_to(OUT.resolve()):
        parser.error("--output-dir must be under outputs/split_training")
    if args.model == "7b" and args.precision == "bf16":
        parser.error("7B requires 4-bit NF4")
    trainer_data = OUT / "preferences" / "trainer_data"
    if args.cv_fold:
        cv_root = OUT.resolve() / "cv"
        arm_root = cv_root / args.cv_fold / f"{args.model}_{args.prompt}"
        if (args.split_file.resolve() != cv_root / "folds.json" or not run_dir.is_relative_to(arm_root)
                or args.pairs.resolve() != arm_root / "pairs.json"):
            parser.error("CV folds, pairs and outputs must use the corresponding outputs/split_training/cv paths")
        if args.model != "3b" or args.precision != CV_PRECISIONS[args.prompt]:
            parser.error("CV requires 3B BF16 for zero-shot and 3B NF4 for few-shot")
        trainer_data = cv_root / "trainer_data"
    pair_bytes, split_bytes = args.pairs.read_bytes(), args.split_file.read_bytes()
    rows = verified_pairs(json.loads(pair_bytes), json.loads(split_bytes), args.prompt, cv_fold=args.cv_fold)
    if args.validate_only:
        scope = args.cv_fold or "dev-only"
        print(f"Verified {len(rows)} {scope} preference pairs.", flush=True)
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    design = {
        "model": args.model,
        "checkpoint": list(MODELS[args.model]),
        "prompt": args.prompt,
        "lr": args.lr,
        "epochs": args.epochs,
        "precision_request": args.precision,
        "pair_count": len(rows),
        "pairs_sha256": sha256(pair_bytes).hexdigest(),
        "split_sha256": sha256(split_bytes).hexdigest(),
        "lora": LORA_SETTINGS,
        "seed": DESIGN["seed"],
        "batch_size": 1,
        "gradient_accumulation": 8,
        "gradient_checkpointing": True,
        "beta": 0.1,
        "save_only_model": True,
        "precompute_disabled_adapter_reference": True,
    }
    if args.cv_fold:
        design["cv_fold"] = args.cv_fold
    design_path, status_path = run_dir / "design.json", run_dir / "status.json"
    if design_path.exists() and json.loads(design_path.read_text(encoding="utf-8")) != design:
        raise ValueError(f"Cannot resume a different experimental design at {run_dir}")
    write_json(design_path, design)
    if not rows:
        write_json(status_path, {"status": "no_pairs", "reason": "No strict preference pairs were available"})
        return 0

    import torch

    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"attempts": {}}
    if status.get("status") == "complete":
        finished_dir = (run_dir / status["checkpoints"][-1]["path"]).parent
        if checkpoint_records(finished_dir, run_dir) == status["checkpoints"]:
            print(f"Training already complete at {run_dir}", flush=True)
            return 0
    order = (args.precision,) if args.precision != "auto" else (
        ("nf4",) if args.model == "7b" else ("bf16", "nf4")
    )
    dataset = preference_dataset(rows, design["pairs_sha256"], trainer_data)
    for precision in order:
        previous = status["attempts"].get(precision, {})
        if precision == "bf16" and len(order) > 1 and previous.get("status") == "oom":
            continue
        status.update(status="running", precision=precision)
        status.pop("reason", None)
        write_json(status_path, status)
        print(f"DPO {args.model} {args.prompt} at {args.lr:g} with {precision} on {len(rows)} pairs", flush=True)
        try:
            result = train_attempt(args, dataset, run_dir, precision)
        except Exception as error:
            status.update(status="failed", reason=f"{type(error).__name__}: {error}")
            write_json(status_path, status)
            raise
        status["attempts"][precision] = result
        status.update(result)
        write_json(status_path, status)
        if result["status"] == "complete":
            return 0
        print(f"{precision} ran out of GPU memory: {result['reason']}", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
