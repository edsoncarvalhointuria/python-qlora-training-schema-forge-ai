import json
from typing import Literal
import pandas as pd
from sklearn import model_selection
import re
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
data_path = Path.cwd().parent / "data"

system_messages = {
    "zod": "Você é um Engenheiro de Software Sênior especialista em TypeScript. Sua tarefa é receber um objeto JSON bruto e gerar o código TypeScript correspondente utilizando a biblioteca 'zod' e estendendo com '@asteasolutions/zod-to-openapi'. Crie schemas modulares, fortemente tipados e inclua descrições e exemplos OpenAPI detalhados baseados nos valores do JSON. Retorne APENAS o código TypeScript, sem formatação markdown (```typescript) e sem explicações textuais.",
    "yaml": "Você é um Arquiteto de Software especialista em documentação de APIs. Converta o JSON de entrada em uma especificação OpenAPI 3.0 em formato YAML. Estruture corretamente os componentes (schemas) e garanta a tipagem rigorosa. Retorne APENAS o código YAML válido, sem comentários adicionais.",
    "json": "Você é um Arquiteto de Dados especialista em modelagem JSON. Sua tarefa é analisar os dados brutos de entrada e gerar um JSON Schema (Draft 2020-12) robusto e perfeitamente validado. Defina rigorosamente os tipos (string, number, boolean, array, object), especifique quais propriedades são obrigatórias e mapeie estruturas aninhadas com precisão. Retorne APENAS o JSON Schema cru, sem formatação markdown (```json) e sem conversas textuais.",
}


login(token=os.getenv("HF_API_KEY"))


def getFineTurningObject(
    type: Literal["zod", "yaml", "json"], user: dict, assistant: dict
):
    return {
        "type": type,
        "system": system_messages[type],
        "user": json.dumps(user),
        "assistant": assistant,
    }


with open(data_path / "fine_turning.jsonl", "a", encoding="utf-8") as final_archive:

    regex = r"([^\\])\\'"
    sub = r"\1\\\'"
    with open(data_path / "zod.jsonl", "r", encoding="utf-8") as zod:
        for i, line in enumerate(zod):
            item = json.loads(re.sub(regex, sub, json.loads(line)))
            obj = getFineTurningObject("zod", item["raw_json"], item["zod_code"])
            formated_obj = json.dumps(obj, ensure_ascii=False)

            final_archive.write(formated_obj + "\n")

    with open(data_path / "json_and_yml.jsonl", "r", encoding="utf-8") as jy:
        for i, line in enumerate(jy):
            item = json.loads(re.sub(regex, sub, json.loads(line)), strict=False)

            if item["openapi_spec"].startswith("{"):
                type = "json"
            else:
                type = "yaml"

            obj = getFineTurningObject(type, item["raw_json"], item["openapi_spec"])
            formated_obj = json.dumps(obj, ensure_ascii=False)
            final_archive.write(formated_obj + "\n")


df = pd.read_json(data_path / "fine_turning.jsonl", lines=True)

df_train, df_test = model_selection.train_test_split(
    df, test_size=0.10, stratify=df["type"], random_state=42
)

df_train.drop(columns=["type"], inplace=True)
df_test.drop(columns=["type"], inplace=True)

df_train.to_json(
    data_path / "train.jsonl", orient="records", lines=True, force_ascii=False
)
df_test.to_json(
    data_path / "test.jsonl", orient="records", lines=True, force_ascii=False
)
