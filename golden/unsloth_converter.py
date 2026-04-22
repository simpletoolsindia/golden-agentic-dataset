"""
Unsloth-Compatible Converter
============================
Converts golden dataset to Unsloth's expected format.

Unsloth expects:
- conversations: list of {from, value} messages
- system: optional system prompt

ChatML format:
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>

Or for tool-calling models, use function call format:
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
<tool_call>
{{"name": "...", "arguments": {{"..."}}}}
</tool_call><|im_end|>
"""

import json
from pathlib import Path
from golden_dataset_v2 import GoldenDatasetGenerator, GoldenValidator, STANDARD_TOOLS, RULES

SYSTEM_PROMPT = """You are an expert AI coding assistant. You have access to tools for reading, editing, and executing code.

Available tools:
{tools}

Rules:
{default_rules}

Follow the tool schemas exactly. Only use tools when needed."""

def format_tools_for_system() -> str:
    lines = []
    for t in STANDARD_TOOLS:
        args = ", ".join(f"{k}: {v.get('type', 'string')}" for k, v in t["input_schema"]["properties"].items())
        lines.append(f"- {t['name']}({args}): {t['description']}")
    return "\n".join(lines)


def convert_to_chatml(sample: dict) -> dict:
    """Convert golden sample to ChatML format for Unsloth."""

    tools_str = format_tools_for_system()
    system = SYSTEM_PROMPT.format(tools=tools_str, default_rules="\n".join(RULES[:5]))

    # Build conversation with tool calls
    messages = [
        {"from": "system", "value": system},
        {"from": "human", "value": sample["instruction"]},
    ]

    # Add assistant tool calls and responses
    for i, tc in enumerate(sample["tool_calls"]):
        tool_name = tc["name"]
        args = json.dumps(tc["arguments"])

        # Format as assistant tool call
        tool_msg = f'<tool_call name="{tool_name}">\n{args}\n</tool_call>'
        messages.append({"from": "gpt", "value": tool_msg})

        # Add tool output if available
        if i < len(sample.get("tool_outputs", [])):
            output = sample["tool_outputs"][i]["output"]
            messages.append({"from": "tool", "value": str(output)})

    # Final assistant response
    final = sample["final_output"]
    final_msg = f'Status: {final["status"]}\n{final["explanation"]}'
    messages.append({"from": "gpt", "value": final_msg})

    return {
        "conversations": messages,
        "system": system,
    }


def convert_to_unsloth_sft(sample: dict) -> dict:
    """Convert to Unsloth SFT format (simpler Q&A pairs)."""

    tools_str = format_tools_for_system()
    system = SYSTEM_PROMPT.format(tools=tools_str, default_rules="\n".join(RULES[:5]))

    # Combine tool calls into a single response
    tool_parts = []
    for i, tc in enumerate(sample["tool_calls"]):
        tool_parts.append(f'Tool: {tc["name"]}\nArgs: {json.dumps(tc["arguments"])}')
        if i < len(sample.get("tool_outputs", [])):
            tool_parts.append(f'Output: {sample["tool_outputs"][i]["output"]}')

    assistant_response = (
        f'Thought: {" → ".join(sample["reasoning"])}\n\n'
        + "\n\n".join(tool_parts)
        + f'\n\nResult: {sample["final_output"]["status"]} - {sample["final_output"]["explanation"]}'
    )

    return {
        "system": system,
        "instruction": sample["instruction"],
        "input": "",
        "output": assistant_response,
    }


def convert_batch(input_path: str, output_path: str, format: str = "chatml"):
    """Convert entire JSONL file to Unsloth format."""

    converter = convert_to_chatml if format == "chatml" else convert_to_unsloth_sft

    count = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue
            sample = json.loads(line)
            converted = converter(sample)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} samples to {format} format")
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="golden_full.jsonl")
    parser.add_argument("--output", type=str, default="golden_unsloth_chatml.jsonl")
    parser.add_argument("--format", choices=["chatml", "sft"], default="chatml")
    args = parser.parse_args()

    count = convert_batch(args.input, args.output, args.format)
    print(f"Output: {args.output}")

    # Validate output
    valid = 0
    with open(args.output) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                json.loads(line)
                valid += 1
            except:
                pass
    print(f"Valid JSONL entries: {valid}/{count}")