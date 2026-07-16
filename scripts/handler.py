from pathlib import Path
from generate_schema import createSchemas

if __name__ == "__main__":
    createSchemas("gemini-3.1-flash-lite", Path.cwd().parent / "data/zod.jsonl")
