#!/usr/bin/env python3
"""
APRIEL-1.6-15B THINKER — POLISH QLoRA TRAINING (ETAP 2, AGGRESSIVE LORA, 12GB VRAM)
=======================================================================

✔ QLoRA 4-bit (nf4)
✔ Bez lm_head
✔ Bez reasoning leakage
✔ Dynamic padding
✔ Gradient checkpointing
✔ Single-GPU friendly (12 GB VRAM)
"""

import os
import sys
import json
import torch
import importlib
import subprocess
import shutil
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# ============================================================================
# BASIC CONFIG
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_URL = "https://huggingface.co/ServiceNow-AI/Apriel-1.6-15b-Thinker"
DEFAULT_LOCAL_DATASET = SCRIPT_DIR / "dataset_part_2_filtered_768.json"


def _normalize_model_id(model_ref: str) -> str:
    """Accept full HF URL or repo id and return repo id."""
    ref = (model_ref or "").strip()
    if ref.startswith("https://huggingface.co/"):
        ref = ref.replace("https://huggingface.co/", "").strip("/")
    return ref or "ServiceNow-AI/Apriel-1.6-15b-Thinker"


MODEL_ID = _normalize_model_id(os.environ.get("BASE_MODEL_ID", DEFAULT_MODEL_URL))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./apriel-pl-lora-part2")
LOCAL_DATASET_PATH = os.environ.get("LOCAL_DATASET_PATH", str(DEFAULT_LOCAL_DATASET))

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "").strip()
HF_DATASET_FILE = os.environ.get("HF_DATASET_FILE", "").strip()
HF_LORA_REPO = os.environ.get("HF_LORA_REPO", "username/apriel-pl-lora")
HF_UPLOAD_INTERVAL = int(os.environ.get("HF_UPLOAD_INTERVAL", "50"))

MAX_LENGTH = 768
BATCH_SIZE = 1
GRAD_ACCUM = 8
EPOCHS = 1
LR = 5e-7

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

SEED = 42

# ============================================================================
# DEPENDENCY CHECK (NO RESTART)
# ============================================================================

def ensure_deps():
    pkgs = [
        ("torch", "torch"),
        ("transformers", "transformers>=4.40"),
        ("peft", "peft>=0.11"),
        ("bitsandbytes", "bitsandbytes>=0.43"),
        ("datasets", "datasets"),
        ("accelerate", "accelerate>=0.29"),
    ]
    for name, pip in pkgs:
        try:
            importlib.import_module(name)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip])

# ============================================================================
# DATASET / HUB HELPERS
# ============================================================================


def download_dataset_from_hub() -> str:
    """Download dataset file from Hugging Face Hub (or reuse cached copy)."""
    print(f"📥 Downloading dataset {HF_DATASET_REPO}/{HF_DATASET_FILE} ...")
    downloaded_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename=HF_DATASET_FILE,
        token=HF_TOKEN or None,
    )
    target = Path(LOCAL_DATASET_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(downloaded_path, target)
    print(f"✅ Dataset saved to {target}")
    return str(target)


def upload_folder_to_hf(path: str, repo_id: str, token: str, commit_message: str):
    """Upload entire folder to Hugging Face Hub."""
    if not token:
        print("⚠️ Skipping upload – HF_TOKEN not provided.")
        return

    api = HfApi(token=token)
    print(f"📤 Uploading {path} to {repo_id} ({commit_message}) ...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    print("✅ Upload finished.")


class HfLoraUploadCallback(TrainerCallback):
    """Uploads checkpoints to HF Hub whenever Trainer saves."""

    def __init__(self, repo_id: str, token: str):
        self.repo_id = repo_id
        self.token = token

    def on_save(self, args, state, control, **kwargs):
        if not self.token:
            return
        checkpoint_folder = kwargs.get("checkpoint_folder")
        if checkpoint_folder:
            upload_folder_to_hf(
                path=checkpoint_folder,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"Checkpoint step {state.global_step}",
            )

# ============================================================================
# DATASET
# ============================================================================

def load_dataset_jsonl(path: str) -> Dataset:
    """Loads either JSONL or JSON-array dataset files into a Dataset."""
    with open(path, "r", encoding="utf-8") as f:
        raw_data = f.read()

    stripped = raw_data.lstrip()
    if not stripped:
        raise ValueError(f"Dataset {path} is empty.")

    try:
        if stripped[0] == "[":
            # Full JSON array
            examples = json.loads(raw_data)
        else:
            # JSONL format (one object per line)
            examples = []
            for line in raw_data.splitlines():
                line = line.strip()
                if not line:
                    continue
                examples.append(json.loads(line))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse dataset {path}: {exc}") from exc

    rows = []
    for ex in examples:
        if "instruction" in ex and "response" in ex:
            rows.append({
                "text": f"{ex['instruction'].strip()}\n\n{ex['response'].strip()}"
            })

    if not rows:
        raise ValueError("Dataset empty or malformed – missing 'instruction'/'response' pairs.")
    return Dataset.from_list(rows)

# ============================================================================
# TOKENIZATION
# ============================================================================

def tokenize(dataset: Dataset, tokenizer):
    def _tok(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            add_special_tokens=True,
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(
        _tok,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

# ============================================================================
# TRAINING
# ============================================================================

def prompt_runtime_inputs():
    """Ask user for model URL and HF token at runtime."""
    global MODEL_ID, HF_TOKEN

    current_model = MODEL_ID or DEFAULT_MODEL_URL
    model_input = input(f"Podaj adres modelu (HF URL lub repo id) [{current_model}]: ").strip()
    if model_input:
        MODEL_ID = _normalize_model_id(model_input)
    else:
        MODEL_ID = _normalize_model_id(current_model)

    if HF_TOKEN:
        token_input = input("Podaj token Hugging Face (ENTER aby użyć ustawionego): ").strip()
        if token_input:
            HF_TOKEN = token_input
    else:
        HF_TOKEN = input("Podaj token Hugging Face (wymagany do uploadu): ").strip()
        if not HF_TOKEN:
            print("⚠️ Nie podano tokenu – upload na HF nie będzie możliwy.")


def resolve_dataset_path() -> str:
    """Return dataset path, download from HF only if local file missing and repo info provided."""
    local_path = Path(LOCAL_DATASET_PATH)
    if local_path.exists():
        print(f"📚 Używam lokalnego zbioru danych: {local_path}")
        return str(local_path)
    if HF_DATASET_REPO and HF_DATASET_FILE:
        return download_dataset_from_hub()
    raise FileNotFoundError(
        f"Nie znaleziono datasetu {local_path}. Podaj HF_DATASET_REPO/FILE lub umieść plik lokalnie."
    )


def main():
    ensure_deps()

    torch.manual_seed(SEED)

    prompt_runtime_inputs()

    # -------------------------
    # Download & load dataset
    # -------------------------
    dataset_path = resolve_dataset_path()
    dataset = load_dataset_jsonl(dataset_path)

    # -------------------------
    # Processor / tokenizer
    # -------------------------
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # Quantization
    # -------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # -------------------------
    # Model
    # -------------------------
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    # -------------------------
    # LoRA (SAFE TARGETS)
    # -------------------------
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # -------------------------
    # Tokenize
    # -------------------------
    tokenized = tokenize(dataset, tokenizer)

    # -------------------------
    # Training data prep
    # -------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=False,
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        weight_decay=0.01,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=1,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
    )

    # -------------------------
    # Collator
    # -------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # -------------------------
    # Trainer
    # -------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[HfLoraUploadCallback(HF_LORA_REPO, HF_TOKEN)],
    )

    # -------------------------
    # Train!
    # -------------------------
    trainer.train()

    # -------------------------
    # Save final adaptor
    # -------------------------
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    upload_folder_to_hf(
        path=OUTPUT_DIR,
        repo_id=HF_LORA_REPO,
        token=HF_TOKEN,
        commit_message="Final checkpoint",
    )

    print("\n✅ TRAINING FINISHED SUCCESSFULLY")
    print(f"Adapter saved to: {OUTPUT_DIR}")

# ============================================================================
# ENTRY
# ============================================================================

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

    main()
