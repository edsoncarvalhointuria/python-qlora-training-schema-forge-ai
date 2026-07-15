from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
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
        target_modules=["c_attn"],
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

    model_name = "gpt2"
    translate = AutoTokenizer.from_pretrained(model_name)
    translate.pad_token = translate.eos_token
    translate.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
    brain = AutoModelForCausalLM.from_pretrained(model_name)
    rules = TrainingArguments(
        output_dir="./meu-modelo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
    )

    trainer = SFTTrainer(
        model=brain,
        train_dataset=dataset["train"].select(range(50)),
        eval_dataset=dataset["test"].select(range(10)),
        args=rules,
        processing_class=translate,
        peft_config=lora,
    )

    trainer.train()
