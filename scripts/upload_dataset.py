from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
data_path = Path.cwd().parent / "data"

archives = {
    "train": data_path / "train.jsonl",
    "test": data_path / "test.jsonl",
}

login(token=os.getenv("HF_API_KEY"))

dataset = load_dataset("json", data_files=archives)
dataset.push_to_hub(
    "edsoncarvalhointuria/erp-schema-forg-dataset",
    commit_message="subindo arquivos atualizados",
)
