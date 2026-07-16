from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
import torch
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
import os
import json

load_dotenv()


def transformTextToMessages(data: dict):
    return {
        "messages": [
            {"role": key, "content": json.dumps(value) if key == "user" else value}
            for key, value in data.items()
        ]
    }


if __name__ == "__main__":
    login(os.getenv("HF_API_KEY"))

    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_dataset("edsoncarvalhointuria/erp-schema-forg-dataset")

    dataset = dataset.map(
        transformTextToMessages,
        num_proc=os.cpu_count(),
        batched=False,
        remove_columns=dataset["train"].column_names,
    )

    model_name = "microsoft/Phi-3-mini-4k-instruct"
    translate = AutoTokenizer.from_pretrained(model_name)
    config_4bits = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    brain = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=config_4bits, dtype=torch.float16
    )
    rules = TrainingArguments(
        output_dir="./resultados",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        num_train_epochs=1,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        save_steps=200,
    )

    trainer = SFTTrainer(
        model=brain,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=rules,
        processing_class=translate,
        peft_config=lora,
        max_seq_length=4096,
    )

    trainer.train()
    trainer.push_to_hub("edsoncarvalhointuria/schema-forg-ai")
