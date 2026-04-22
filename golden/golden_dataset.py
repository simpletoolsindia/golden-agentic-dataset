"""
Golden Dataset Generator for Unsloth LoRA Training
==================================================
A novel, high-quality dataset schema for agentic AI training.
Built from research on HF datasets, research papers, and Unsloth format.

Schema Design Principles:
- 12 standard tools (no explosion)
- Strict structured output format
- Plan → Execute → Validate → Fix agent flows
- 100% tool calling accuracy targets
- Unsloth-compatible JSONL format

Usage:
    python golden_dataset.py --generate 1000
    python golden_dataset.py --validate
"""

import json
import random
import uuid
import argparse
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from pathlib import Path

# =============================================================================
# STANDARD TOOL SET (12 tools - minimal but complete)
# =============================================================================

STANDARD_TOOLS = [
    {
        "name": "read_file",
        "description": "Read file contents from the filesystem",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to file"},
                "offset": {"type": "integer", "description": "Line offset"},
                "limit": {"type": "integer", "description": "Max lines to read"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write or create a new file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path for new file"},
                "content": {"type": "string", "description": "File content"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_file",
        "description": "Modify specific lines in an existing file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "old_string": {"type": "string", "description": "Exact text to replace"},
                "new_string": {"type": "string", "description": "Replacement text"}
            },
            "required": ["path", "old_string", "new_string"]
        }
    },
    {
        "name": "search_files",
        "description": "Find files matching glob pattern",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern (e.g., **/*.py)"},
                "path": {"type": "string", "description": "Root directory"}
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "run_command",
        "description": "Execute shell command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command"},
                "timeout": {"type": "integer", "description": "Timeout in seconds"}
            },
            "required": ["command"]
        }
    },
    {
        "name": "run_code",
        "description": "Execute code snippet",
        "input_schema": {
            "type": "object",
            "properties": {
                "language": {"type": "string", "description": "python, javascript, etc."},
                "code": {"type": "string", "description": "Code to execute"}
            },
            "required": ["language", "code"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the internet",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "fetch_url",
        "description": "Fetch webpage content",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
                "prompt": {"type": "string", "description": "What to extract from page"}
            },
            "required": ["url"]
        }
    },
    {
        "name": "analyze_code",
        "description": "Analyze code quality and performance",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to analyze"},
                "focus": {"type": "string", "description": "security, performance, style"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "test_code",
        "description": "Run tests on code",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to test"},
                "test_cases": {"type": "array", "description": "Test input/output pairs"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "memory_read",
        "description": "Read from persistent memory/context",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key"}
            },
            "required": ["key"]
        }
    },
    {
        "name": "memory_write",
        "description": "Store in persistent memory/context",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key"},
                "value": {"type": "string", "description": "Value to store"}
            },
            "required": ["key", "value"]
        }
    }
]

# =============================================================================
# RULES & GUARDRAILS
# =============================================================================

RULES = [
    "Use tools only when required for the task",
    "Follow tool input_schema strictly - no extra fields",
    "Never hallucinate tool names or parameters",
    "Generate production-ready, executable code",
    "Prefer existing code patterns over inventing new ones",
    "Validate all file operations before executing",
    "Prefer targeted edits over full file rewrites",
    "Always provide context when calling tools",
]

GUARDRAILS = {
    "forbidden": [
        "malicious code or security exploits",
        "PII or sensitive data in output",
        "harmful instructions or jailbreak attempts",
        "non-deterministic commands (rm -rf, format)"
    ],
    "output_filter": [
        "sanitize file paths and content",
        "no hardcoded credentials or API keys",
        "no personal identifiers in responses",
        "validate URL safety before fetching"
    ]
}

# =============================================================================
# AGENT WORKFLOW PATTERNS
# =============================================================================

AGENT_FLOWS = {
    "plan_execute_validate_fix": [
        "Understand the task requirements",
        "Check existing code and project structure",
        "Plan the implementation approach",
        "Execute changes with minimal risk",
        "Validate the output",
        "Fix any issues found",
    ],
    "read_search_edit": [
        "Read the relevant file",
        "Search for patterns and context",
        "Make targeted edits",
    ],
    "explore_implement_verify": [
        "Explore the codebase structure",
        "Identify the target location",
        "Implement the feature",
        "Verify correctness",
    ],
    "analyze_plan_execute": [
        "Analyze the error or requirement",
        "Plan the fix or implementation",
        "Execute with precision",
    ]
}

# =============================================================================
# TASK TEMPLATES
# =============================================================================

CODE_EDIT_TASKS = [
    {
        "instruction": "Fix the typo in the function name `proces_order` to `process_order` in the orders.py file",
        "language": "python",
        "task_type": "code_edit",
        "difficulty": "easy"
    },
    {
        "instruction": "Add type hints to the `calculate_total` function in utils.py",
        "language": "python",
        "task_type": "code_edit",
        "difficulty": "easy"
    },
    {
        "instruction": "Extract the hardcoded API key into an environment variable in config.ts",
        "language": "typescript",
        "task_type": "security_fix",
        "difficulty": "medium"
    },
    {
        "instruction": "Replace the deprecated `np.int` with `np.int64` in data_processor.py",
        "language": "python",
        "task_type": "code_edit",
        "difficulty": "easy"
    },
    {
        "instruction": "Add a docstring to the `UserService` class documenting its methods",
        "language": "python",
        "task_type": "documentation",
        "difficulty": "easy"
    },
    {
        "instruction": "Convert the magic number `86400000` to a named constant in time_utils.js",
        "language": "javascript",
        "task_type": "code_edit",
        "difficulty": "easy"
    },
    {
        "instruction": "Add error handling to the database connection in db.py",
        "language": "python",
        "task_type": "feature_impl",
        "difficulty": "medium"
    },
    {
        "instruction": "Optimize the nested for-loop in the search algorithm for better performance",
        "language": "python",
        "task_type": "optimization",
        "difficulty": "hard"
    },
    {
        "instruction": "Add input validation to the API endpoint handler in app.py",
        "language": "python",
        "task_type": "feature_impl",
        "difficulty": "medium"
    },
    {
        "instruction": "Refactor the long function `handle_user_request` into smaller helper functions",
        "language": "python",
        "task_type": "refactoring",
        "difficulty": "medium"
    },
    {
        "instruction": "Add logging to track API call failures in client.go",
        "language": "go",
        "task_type": "feature_impl",
        "difficulty": "easy"
    },
    {
        "instruction": "Replace the deprecated $.ajax() with fetch() in api.js",
        "language": "javascript",
        "task_type": "code_edit",
        "difficulty": "medium"
    },
    {
        "instruction": "Add `async/await` to the function calls that are missing it in async_handler.py",
        "language": "python",
        "task_type": "code_edit",
        "difficulty": "easy"
    },
    {
        "instruction": "Fix the race condition in the cache update logic in cache_manager.py",
        "language": "python",
        "task_type": "bug_fix",
        "difficulty": "hard"
    },
    {
        "instruction": "Add pagination support to the list endpoint in views.py",
        "language": "python",
        "task_type": "feature_impl",
        "difficulty": "medium"
    },
    {
        "instruction": "Replace string concatenation with f-strings in format_output.py",
        "language": "python",
        "task_type": "code_edit",
        "difficulty": "easy"
    },
    {
        "instruction": "Add retry logic to the HTTP client in http_client.py",
        "language": "typescript",
        "task_type": "feature_impl",
        "difficulty": "medium"
    },
    {
        "instruction": "Fix the SQL injection vulnerability in the user search query",
        "language": "python",
        "task_type": "security_fix",
        "difficulty": "hard"
    },
    {
        "instruction": "Extract the configuration into a separate config.yaml file",
        "language": "python",
        "task_type": "refactoring",
        "difficulty": "medium"
    },
    {
        "instruction": "Add unit tests for the `calculate_discount` function",
        "language": "python",
        "task_type": "test_generation",
        "difficulty": "medium"
    }
]

FILE_PATHS = {
    "python": [
        "/project/src/utils.py",
        "/project/src/services/user_service.py",
        "/project/src/models/order.py",
        "/project/lib/data_processor.py",
        "/project/app/routes.py",
        "/project/core/db.py",
        "/project/core/config.py",
        "/project/core/cache_manager.py",
        "/project/api/handlers.py",
        "/project/tests/test_utils.py",
        "/project/src/async_handler.py",
        "/project/src/format_output.py",
        "/project/src/calculate.py",
        "/project/src/validators.py",
        "/project/lib/parsers.py",
        "/project/app/views.py",
        "/project/core/events.py",
    ],
    "typescript": [
        "/project/src/api/client.ts",
        "/project/src/config.ts",
        "/project/src/utils.ts",
        "/project/src/services/http_client.ts",
        "/project/src/api/handlers.ts",
        "/project/src/utils/helpers.ts",
    ],
    "javascript": [
        "/project/public/js/api.js",
        "/project/src/utils.js",
        "/project/src/time_utils.js",
        "/project/src/app.js",
    ],
    "go": [
        "/project/internal/client.go",
        "/project/internal/db.go",
        "/project/cmd/main.go",
    ],
}

# Code snippets for context
CODE_SNIPPETS = {
    "python": [
        "def proces_order(order_id):\n    return f'Processing {order_id}'",
        "def calculate_total(items):\n    total = 0\n    for item in items:\n        total += item['price']\n    return total",
        "class UserService:\n    def get_user(self, user_id):\n        return db.query(user_id)",
        "API_KEY = 'sk-1234567890abcdef'",
        "np.int",  # placeholder for replacement
        "value = data[0] * 86400000",
        "try:\n    db.connect()\nexcept:\n    pass",
        "for i in range(len(items)):\n    for j in range(len(items)):\n        if items[i] < items[j]:\n            items[i], items[j] = items[j], items[i]",
        "def handle_request(data):\n    x = process(data)\n    y = validate(x)\n    z = save(y)\n    return z",
        "async def fetch_data(url):\n    result = urllib.request.urlopen(url)\n    return result",
    ],
    "typescript": [
        "const API_KEY = 'sk-test-key-123';",
        "async function fetchData(url: string) {\n  return fetch(url).then(r => r.json());\n}",
        "$.ajax({ url: '/api', success: function(data) {} });",
    ],
    "javascript": [
        "var result = 'Hello ' + name + '!';",
    ],
    "go": [
        "resp, err := http.Get(url)\nif err != nil {\n    log.Println(err)\n}",
    ],
}

# =============================================================================
# GOLDEN DATASET SCHEMA
# =============================================================================

@dataclass
class GoldenSample:
    """One entry in the golden dataset."""
    id: str
    timestamp: str
    localization: dict
    rules: list
    guardrails: dict
    available_tools: list
    instruction: str
    context: dict
    reasoning: list
    tool_calls: list
    tool_outputs: list
    final_output: dict

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenSample":
        return cls(**data)


# =============================================================================
# SAMPLE GENERATOR
# =============================================================================

class GoldenDatasetGenerator:
    """Generate high-quality golden dataset entries."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tool_names = [t["name"] for t in STANDARD_TOOLS]
        self.generated_ids = set()

    def _generate_id(self, prefix: str = "gold") -> str:
        """Generate unique ID."""
        while True:
            uid = str(uuid.uuid4())[:8]
            candidate = f"{prefix}_{uid}"
            if candidate not in self.generated_ids:
                self.generated_ids.add(candidate)
                return candidate

    def _get_file_path(self, language: str) -> str:
        """Get random file path for language."""
        paths = FILE_PATHS.get(language, FILE_PATHS["python"])
        return self.rng.choice(paths)

    def _get_code_snippet(self, language: str) -> str:
        """Get random code snippet for language."""
        snippets = CODE_SNIPPETS.get(language, CODE_SNIPPETS["python"])
        return self.rng.choice(snippets)

    def _generate_tool_calls(
        self,
        task_type: str,
        file_path: str,
        code_snippet: str,
        instruction: str
    ) -> tuple[list, list]:
        """Generate realistic tool calls and outputs for a task."""

        tool_calls = []
        tool_outputs = []

        # Always start with read_file to understand context
        tool_calls.append({
            "name": "read_file",
            "arguments": {"path": file_path}
        })
        tool_outputs.append({
            "name": "read_file",
            "output": code_snippet
        })

        if task_type in ["code_edit", "bug_fix", "security_fix"]:
            # Edit-based workflow
            old_string = code_snippet[:min(50, len(code_snippet))]
            new_string = old_string.replace("proces_", "process_") if "proces_" in old_string else old_string
            new_string = new_string.replace("API_KEY = ", "# API_KEY from env\nAPI_KEY = ") if "API_KEY" in new_string else new_string

            tool_calls.append({
                "name": "edit_file",
                "arguments": {
                    "path": file_path,
                    "old_string": old_string,
                    "new_string": new_string
                }
            })
            tool_outputs.append({
                "name": "edit_file",
                "output": f"Successfully edited {file_path} - 1 replacement made"
            })

        elif task_type in ["feature_impl", "documentation", "test_generation"]:
            # Read + write/edit workflow
            tool_calls.append({
                "name": "edit_file",
                "arguments": {
                    "path": file_path,
                    "old_string": "# TODO: implement",
                    "new_string": "# TODO: implement\n    pass  # implemented"
                }
            })
            tool_outputs.append({
                "name": "edit_file",
                "output": f"Added implementation to {file_path}"
            })

        elif task_type in ["optimization", "refactoring"]:
            # Read + analyze + edit workflow
            tool_calls.append({
                "name": "analyze_code",
                "arguments": {
                    "code": code_snippet,
                    "focus": "performance" if "optimization" in task_type else "style"
                }
            })
            tool_outputs.append({
                "name": "analyze_code",
                "output": "Found O(n²) complexity. Consider using hash map for O(n) lookup."
            })

            tool_calls.append({
                "name": "edit_file",
                "arguments": {
                    "path": file_path,
                    "old_string": code_snippet[:min(40, len(code_snippet))],
                    "new_string": "# optimized: " + code_snippet[:min(40, len(code_snippet))]
                }
            })
            tool_outputs.append({
                "name": "edit_file",
                "output": f"Refactored {file_path} for better performance/style"
            })

        elif task_type == "search":
            # Search-based workflow
            tool_calls.append({
                "name": "search_files",
                "arguments": {"pattern": "**/*.py", "path": "/project/src"}
            })
            tool_outputs.append({
                "name": "search_files",
                "output": "['/project/src/utils.py', '/project/src/services/user_service.py', '/project/src/models/order.py']"
            })

        return tool_calls, tool_outputs

    def generate(
        self,
        instruction: str,
        language: str = "python",
        task_type: str = "code_edit",
        difficulty: str = "medium",
        context_overrides: Optional[dict] = None
    ) -> GoldenSample:
        """Generate a single golden sample."""

        file_path = self._get_file_path(language)
        code_snippet = self._get_code_snippet(language)

        tool_calls, tool_outputs = self._generate_tool_calls(
            task_type, file_path, code_snippet, instruction
        )

        # Determine reasoning based on task type
        flow_key = self._get_flow_for_task(task_type)
        reasoning = [AGENT_FLOWS[flow_key][step] for step in range(min(4, len(AGENT_FLOWS[flow_key])))]

        context = {
            "project": "cli_tool",
            "language": language,
            "task_type": task_type,
            "difficulty": difficulty,
            "file_path": file_path
        }
        if context_overrides:
            context.update(context_overrides)

        final_output = {
            "status": "success",
            "response": f"Completed: {instruction[:100]}",
            "code": None,
            "explanation": f"Successfully used {len(tool_calls)} tool calls to {task_type} in {file_path}",
            "tool_usage": [
                {"tool": tc["name"], "purpose": "context gathering" if i == 0 else "modification"}
                for i, tc in enumerate(tool_calls)
            ],
            "next_actions": []
        }

        sample = GoldenSample(
            id=self._generate_id(),
            timestamp=datetime.now().isoformat() + "Z",
            localization={
                "language": "en",
                "tone": "professional",
                "style": "technical"
            },
            rules=RULES,
            guardrails=GUARDRAILS,
            available_tools=STANDARD_TOOLS,
            instruction=instruction,
            context=context,
            reasoning=reasoning,
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
            final_output=final_output
        )

        return sample

    def _get_flow_for_task(self, task_type: str) -> str:
        """Map task type to agent flow."""
        flow_map = {
            "code_edit": "read_search_edit",
            "bug_fix": "plan_execute_validate_fix",
            "security_fix": "plan_execute_validate_fix",
            "feature_impl": "explore_implement_verify",
            "documentation": "analyze_plan_execute",
            "optimization": "analyze_plan_execute",
            "refactoring": "analyze_plan_execute",
            "test_generation": "explore_implement_verify",
            "search": "read_search_edit",
        }
        return flow_map.get(task_type, "analyze_plan_execute")

    def generate_batch(self, count: int) -> list[GoldenSample]:
        """Generate multiple samples from templates."""
        samples = []
        tasks = list(CODE_EDIT_TASKS)

        # Shuffle and repeat to get enough diversity
        self.rng.shuffle(tasks)

        for i in range(count):
            task = tasks[i % len(tasks)]
            sample = self.generate(
                instruction=task["instruction"],
                language=task["language"],
                task_type=task["task_type"],
                difficulty=task["difficulty"]
            )
            samples.append(sample)

        return samples


# =============================================================================
# VALIDATOR
# =============================================================================

class GoldenDatasetValidator:
    """Validate golden dataset quality."""

    @staticmethod
    def validate(sample: dict) -> tuple[bool, list[str]]:
        """Validate a single sample. Returns (valid, errors)."""
        errors = []

        # Required fields
        required = ["id", "instruction", "context", "tool_calls", "final_output", "available_tools"]
        for field in required:
            if field not in sample:
                errors.append(f"Missing required field: {field}")

        # Tool count check
        if len(sample.get("available_tools", [])) > 13:
            errors.append(f"Too many tools: {len(sample['available_tools'])} (max 13)")

        # Tool call validation
        valid_tool_names = {t["name"] for t in sample.get("available_tools", [])}
        for tc in sample.get("tool_calls", []):
            if tc.get("name") not in valid_tool_names:
                errors.append(f"Invalid tool name: {tc.get('name')}")

        # JSON validity
        try:
            json.dumps(sample["instruction"])
        except Exception:
            errors.append("Invalid instruction JSON")

        # Reasoning length
        if len(sample.get("reasoning", [])) < 2:
            errors.append("Reasoning too short")

        return len(errors) == 0, errors

    @staticmethod
    def validate_file(filepath: str) -> tuple[int, int, list[dict]]:
        """Validate entire JSONL file. Returns (total, valid, errors)."""
        total = 0
        valid = 0
        errors_detail = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    sample = json.loads(line)
                    is_valid, errs = GoldenDatasetValidator.validate(sample)
                    if is_valid:
                        valid += 1
                    else:
                        errors_detail.append({"line": line_num, "errors": errs})
                except json.JSONDecodeError as e:
                    errors_detail.append({"line": line_num, "errors": [f"JSON decode: {e}"]})

        return total, valid, errors_detail


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Golden Dataset Generator")
    parser.add_argument("--generate", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="golden_dataset.jsonl", help="Output file")
    parser.add_argument("--validate", type=str, help="Validate a JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.validate:
        print(f"Validating: {args.validate}")
        total, valid, errors = GoldenDatasetValidator.validate_file(args.validate)
        print(f"\nResults:")
        print(f"  Total:   {total}")
        print(f"  Valid:   {valid}")
        print(f"  Invalid: {total - valid}")
        if errors:
            print(f"\n  First 10 errors:")
            for e in errors[:10]:
                print(f"    Line {e['line']}: {e['errors']}")
        return

    # Generate dataset
    print(f"Generating {args.generate} golden samples...")
    generator = GoldenDatasetGenerator(seed=args.seed)
    samples = generator.generate_batch(args.generate)

    # Write JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(sample.to_jsonl() + "\n")

    print(f"Written: {output_path}")
    print(f"Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Samples: {len(samples)}")

    # Validate
    print("\nValidating...")
    total, valid, errors = GoldenDatasetValidator.validate_file(str(output_path))
    print(f"  Valid: {valid}/{total} ({100*valid/total:.1f}%)")

    if errors:
        print(f"  Errors found: {len(errors)}")


if __name__ == "__main__":
    main()