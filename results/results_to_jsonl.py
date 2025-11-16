import json
import re

def count_whitespace(s: str) -> int:
    """Count whitespace characters in expected.
       If expected is empty or '""', return 0."""
    if s.strip() == "" or s.strip() == '""':
        return 0
    return sum(1 for c in s if c.isspace())

def escape_newlines(s: str) -> str:
    """Convert real newline characters into literal '\\n'."""
    return s.replace("\n", "\\n")

def txt_to_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split blocks by "Q:" at the start of a line
    blocks = re.split(r"\n(?=Q:)", text.strip())
    data = []

    for block in blocks:
        q_match = re.search(r"Q:\s*(.*)", block)
        a_match = re.search(r"A:\s*((?:.|\n)*?)(?=\nExpected:|$)", block)
        exp_match = re.search(r"Expected:\s*(.*)", block)

        if not q_match:
            continue

        question = q_match.group(1).strip()

        # Extract answer including multiple lines
        answer_raw = a_match.group(1).rstrip() if a_match else ""
        answer = escape_newlines(answer_raw)

        expected = exp_match.group(1).strip() if exp_match else ""

        columns = count_whitespace(expected)

        data.append({
            "question": question,
            "columns": columns,
            "gold": expected,
            "predicted": answer
        })

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as out:
        for item in data:
            out.write(json.dumps(item, ensure_ascii=False) + "\n")


# Example usage:
txt_to_jsonl(
    "qwen3B-tpch-small-3-epochs.txt",
    "qwen3B-tpch-small-3-epochs.jsonl"
)
