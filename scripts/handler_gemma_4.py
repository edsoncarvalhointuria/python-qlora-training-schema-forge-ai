from pathlib import Path

from generate_schema import createSchemas

if __name__ == "__main__":
    createSchemas("gemma-4-26b-a4b-it", Path.cwd().parent / "data/zod_gemma.jsonl")
