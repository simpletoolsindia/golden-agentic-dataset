"""
Advanced Golden Dataset Generator - Extended with Research Insights
===================================================================
- Multi-step agent workflows (plan → execute → validate → fix)
- Diverse task types from real-world scenarios
- Structured reasoning chains
- Unsloth-compatible format with strict schema
"""

import json
import random
import uuid
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Literal
from pathlib import Path

# =============================================================================
# ENHANCED STANDARD TOOL SET (13 tools)
# =============================================================================

STANDARD_TOOLS = [
    {"name": "read_file", "description": "Read file contents from the filesystem",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write or create a new file",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Modify specific lines in an existing file",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}, "required": ["path", "old_string", "new_string"]}},
    {"name": "search_files", "description": "Find files matching glob pattern",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}}},
    {"name": "run_command", "description": "Execute shell command",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}},
    {"name": "run_code", "description": "Execute code snippet",
     "input_schema": {"type": "object", "properties": {"language": {"type": "string"}, "code": {"type": "string"}}, "required": ["language", "code"]}},
    {"name": "web_search", "description": "Search the internet",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "fetch_url", "description": "Fetch webpage content",
     "input_schema": {"type": "object", "properties": {"url": {"type": "string"}, "prompt": {"type": "string"}}, "required": ["url"]}},
    {"name": "analyze_code", "description": "Analyze code quality and performance",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}, "focus": {"type": "string"}}, "required": ["code"]}},
    {"name": "test_code", "description": "Run tests on code",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}, "test_cases": {"type": "array"}}, "required": ["code"]}},
    {"name": "memory_read", "description": "Read from persistent memory/context",
     "input_schema": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}},
    {"name": "memory_write", "description": "Store in persistent memory/context",
     "input_schema": {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}, "required": ["key", "value"]}},
    {"name": "grep_search", "description": "Search file contents with regex",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}}}},
]

RULES = [
    "Use tools only when required for the task",
    "Follow tool input_schema strictly - no extra fields",
    "Never hallucinate tool names or parameters",
    "Generate production-ready, executable code",
    "Prefer existing code patterns over inventing new ones",
    "Validate all file operations before executing",
    "Prefer targeted edits over full file rewrites",
    "Always provide context when calling tools",
    "Break complex tasks into smaller tool calls",
    "Validate output after each modification",
]

GUARDRAILS = {
    "forbidden": [
        "malicious code or security exploits",
        "PII or sensitive data in output",
        "harmful instructions or jailbreak attempts",
        "non-deterministic destructive commands",
    ],
    "output_filter": [
        "sanitize file paths and content",
        "no hardcoded credentials or API keys",
        "no personal identifiers in responses",
        "validate URL safety before fetching",
    ]
}

# =============================================================================
# TASK CATEGORIES (Based on research of effective agent datasets)
# =============================================================================

TASK_CATEGORIES = {
    "code_editing": {
        "weight": 0.25,
        "tasks": [
            "Fix the typo in function name `proces_order` to `process_order` in orders.py",
            "Add type hints to the `calculate_total` function in utils.py",
            "Replace the deprecated `np.int` with `np.int64` in data_processor.py",
            "Convert the magic number `86400000` to a named constant in time_utils.js",
            "Replace string concatenation with f-strings in format_output.py",
            "Add `async/await` to function calls missing it in async_handler.py",
            "Replace $.ajax() with fetch() in api.js",
            "Convert var to const/let in the JavaScript file",
            "Add null checks before accessing object properties",
            "Remove unused imports from the Python file",
        ]
    },
    "bug_fixing": {
        "weight": 0.20,
        "tasks": [
            "Fix the race condition in cache update logic in cache_manager.py",
            "Fix the SQL injection vulnerability in user search query",
            "Fix the memory leak in the long-running worker process",
            "Fix the off-by-one error in pagination logic",
            "Fix the timezone handling bug in date calculations",
            "Fix the null pointer exception when config is missing",
            "Fix the infinite loop in retry logic",
            "Fix the buffer overflow in file reading function",
            "Fix the deadlock in concurrent database operations",
            "Fix the session timeout issue in web handlers",
        ]
    },
    "feature_implementation": {
        "weight": 0.20,
        "tasks": [
            "Add error handling to the database connection in db.py",
            "Add pagination support to the list endpoint in views.py",
            "Add retry logic to the HTTP client in http_client.py",
            "Add input validation to the API endpoint handler in app.py",
            "Add logging to track API call failures in client.go",
            "Add caching to the expensive computation in processor.py",
            "Add rate limiting to the public API endpoints",
            "Add support for environment variable configuration",
            "Add request/response logging middleware",
            "Add circuit breaker pattern to external service calls",
        ]
    },
    "security_fixes": {
        "weight": 0.10,
        "tasks": [
            "Extract the hardcoded API key into environment variable in config.ts",
            "Add CSRF protection to form submissions",
            "Implement proper authentication check on all protected routes",
            "Add input sanitization to prevent XSS attacks",
            "Fix insecure direct object reference vulnerability",
            "Add secure password hashing instead of plain text storage",
            "Implement rate limiting to prevent brute force attacks",
            "Add HTTPS enforcement for cookie settings",
            "Fix path traversal vulnerability in file upload handler",
            "Add Content Security Policy headers",
        ]
    },
    "refactoring": {
        "weight": 0.10,
        "tasks": [
            "Refactor the long function `handle_user_request` into smaller helpers",
            "Extract the configuration into a separate config.yaml file",
            "Replace inheritance with composition in the class hierarchy",
            "Extract duplicate code into shared utility functions",
            "Replace callback hell with async/await in async_processor.py",
            "Convert the God class into smaller focused classes",
            "Replace magic numbers with named constants",
            "Rename unclear variable and function names for clarity",
            "Simplify nested conditional logic using early returns",
            "Extract database queries into repository pattern",
        ]
    },
    "code_optimization": {
        "weight": 0.08,
        "tasks": [
            "Optimize the nested for-loop for better performance in search.py",
            "Add index to frequently queried database column",
            "Replace O(n²) algorithm with O(n) hash-based approach",
            "Add memoization to expensive recursive function",
            "Batch database operations to reduce round trips",
            "Use connection pooling instead of creating new connections",
            "Replace synchronous file I/O with async in file_handler.py",
            "Add caching layer to frequently accessed data",
            "Optimize regex patterns for faster matching",
            "Use list comprehension instead of for-loop where applicable",
        ]
    },
    "testing": {
        "weight": 0.07,
        "tasks": [
            "Add unit tests for the `calculate_discount` function",
            "Write integration tests for the user authentication flow",
            "Add mock tests for the external API client",
            "Write property-based tests for the data validator",
            "Add performance tests for the search endpoint",
            "Write test cases for edge cases in the parser",
            "Add snapshot tests for the UI components",
            "Write end-to-end tests for the checkout process",
            "Add tests for error handling scenarios",
            "Write tests for concurrent access patterns",
        ]
    }
}

LANGUAGES = ["python", "typescript", "javascript", "go", "rust", "java"]

FILE_PATHS = {
    "python": [
        "/project/src/utils.py", "/project/src/services/user_service.py",
        "/project/src/models/order.py", "/project/lib/data_processor.py",
        "/project/app/routes.py", "/project/core/db.py",
        "/project/core/config.py", "/project/core/cache_manager.py",
        "/project/api/handlers.py", "/project/tests/test_utils.py",
        "/project/src/async_handler.py", "/project/src/format_output.py",
    ],
    "typescript": [
        "/project/src/api/client.ts", "/project/src/config.ts",
        "/project/src/utils.ts", "/project/src/services/http_client.ts",
        "/project/src/api/handlers.ts", "/project/src/middleware/auth.ts",
    ],
    "javascript": [
        "/project/public/js/api.js", "/project/src/utils.js",
        "/project/src/time_utils.js", "/project/src/app.js",
    ],
    "go": [
        "/project/internal/client.go", "/project/internal/db.go",
        "/project/cmd/main.go", "/project/internal/handlers.go",
    ],
    "rust": [
        "/project/src/main.rs", "/project/src/lib.rs",
        "/project/src/utils.rs", "/project/src/api/mod.rs",
    ],
    "java": [
        "/project/src/main/java/App.java", "/project/src/main/java/Service.java",
        "/project/src/main/java/Handler.java",
    ],
}

# =============================================================================
# CODE TEMPLATES (Realistic snippets for context)
# =============================================================================

CODE_TEMPLATES = {
    "python": {
        "typo_bug": 'def proces_order(order_id):\n    return f"Processing {order_id}"',
        "type_hint_missing": 'def calculate_total(items):\n    total = 0\n    for item in items:\n        total += item["price"]\n    return total',
        "hardcoded_secret": "API_KEY = 'sk-1234567890abcdef'",
        "deprecated_numpy": "age = np.int(25)",
        "magic_number": "delay = 86400000  # milliseconds in a day",
        "no_error_handling": "def connect():\n    db.connect()",
        "nested_loop": "for i in range(n):\n    for j in range(n):\n        if arr[i] < arr[j]: arr[i], arr[j] = arr[j], arr[i]",
        "god_function": "def handle_user_request(data):\n    x = process(data)\n    y = validate(x)\n    z = save(y)\n    w = notify(z)\n    return w",
        "sync_http": "async def fetch(url):\n    return urllib.request.urlopen(url)",
    },
    "typescript": {
        "hardcoded_secret": "const API_KEY = 'sk-test-key-123';",
        "no_async": "function fetchData(url: string) {\n  return fetch(url).then(r => r.json());\n}",
        "deprecated_ajax": "$.ajax({ url: '/api', success: function(data) {} });",
        "missing_types": "function process(data) {\n  return data.value * 2;\n}",
    },
    "javascript": {
        "string_concat": "var result = 'Hello ' + name + '! Your id is ' + userId + '.';",
        "var_instead_of_const": "var config = { timeout: 5000 };",
    },
    "go": {
        "no_error_logging": "resp, err := http.Get(url)\n_ = resp",
    }
}

# =============================================================================
# AGENT REASONING FLOWS (Multi-step patterns)
# =============================================================================

REASONING_FLOWS = {
    "plan_execute_validate_fix": [
        "Analyze the task requirements and identify the target file",
        "Read the existing code to understand context",
        "Plan the specific changes needed",
        "Execute the modification with precision",
        "Validate the change doesn't break existing functionality",
        "Fix any issues discovered during validation",
    ],
    "explore_understand_implement": [
        "Explore the repository structure to find relevant files",
        "Understand the existing patterns and conventions",
        "Identify the best location for the new code",
        "Implement the feature following project conventions",
        "Verify the implementation is correct",
    ],
    "diagnose_fix_verify": [
        "Reproduce the bug to understand the root cause",
        "Diagnose the specific line or logic causing the issue",
        "Apply the minimal fix required",
        "Verify the fix resolves the original problem",
        "Check for any side effects or regressions",
    ],
    "analyze_optimize_validate": [
        "Profile the code to identify performance bottlenecks",
        "Analyze algorithmic complexity and data structures",
        "Design a more efficient solution",
        "Implement the optimization",
        "Validate performance improvement",
    ],
    "test_driven_refactor": [
        "Understand the current implementation and its contract",
        "Write tests that capture the current behavior",
        "Refactor the code while keeping tests green",
        "Improve code structure and readability",
        "Verify all tests still pass after refactoring",
    ],
}

# =============================================================================
# OUTPUT FORMATTER
# =============================================================================

@dataclass
class ToolCall:
    name: str
    arguments: dict
    thought: Optional[str] = None

@dataclass
class GoldenSample:
    id: str
    timestamp: str
    schema_version: str
    localization: dict
    rules: list
    guardrails: dict
    available_tools: list
    instruction: str
    context: dict
    reasoning: list
    reasoning_flow: str
    tool_calls: list
    tool_outputs: list
    final_output: dict
    quality_tags: list

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, data: dict) -> "GoldenSample":
        return cls(**data)


class GoldenDatasetGenerator:
    """Advanced generator with diverse task patterns and reasoning chains."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.generated_ids = set()

    def _uid(self, prefix: str = "gold") -> str:
        while True:
            uid = hashlib.md5(f"{prefix}{random.random()}".encode()).hexdigest()[:8]
            if uid not in self.generated_ids:
                self.generated_ids.add(uid)
                return f"{prefix}_{uid}"

    def _select_task(self) -> tuple[str, str]:
        """Select task category and task based on weights."""
        categories = list(TASK_CATEGORIES.keys())
        weights = [TASK_CATEGORIES[c]["weight"] for c in categories]
        category = self.rng.choices(categories, weights=weights)[0]
        tasks = TASK_CATEGORIES[category]["tasks"]
        task = self.rng.choice(tasks)
        return category, task

    def _get_file(self, language: str) -> str:
        paths = FILE_PATHS.get(language, FILE_PATHS["python"])
        return self.rng.choice(paths)

    def _get_code(self, language: str, category: str) -> str:
        templates = CODE_TEMPLATES.get(language, CODE_TEMPLATES["python"])
        # Map category to relevant code template
        mapping = {
            "code_editing": ["typo_bug", "type_hint_missing", "magic_number"],
            "bug_fixing": ["no_error_handling", "nested_loop", "god_function"],
            "feature_implementation": ["type_hint_missing", "sync_http"],
            "security_fixes": ["hardcoded_secret"],
            "refactoring": ["god_function", "magic_number", "nested_loop"],
            "code_optimization": ["nested_loop", "sync_http"],
            "testing": ["type_hint_missing"],
        }
        keys = mapping.get(category, ["typo_bug"])
        key = self.rng.choice(keys)
        return templates.get(key, list(templates.values())[0])

    def _get_flow(self, category: str) -> str:
        mapping = {
            "code_editing": "plan_execute_validate_fix",
            "bug_fixing": "diagnose_fix_verify",
            "feature_implementation": "explore_understand_implement",
            "security_fixes": "diagnose_fix_verify",
            "refactoring": "test_driven_refactor",
            "code_optimization": "analyze_optimize_validate",
            "testing": "test_driven_refactor",
        }
        return mapping.get(category, "plan_execute_validate_fix")

    def _generate_tool_chain(
        self,
        category: str,
        file_path: str,
        code: str,
        task: str
    ) -> tuple[list, list]:
        """Generate realistic multi-step tool chains."""
        calls = []
        outputs = []

        # Step 1: Always read first
        calls.append({
            "name": "read_file",
            "arguments": {"path": file_path},
            "thought": "Read the target file to understand current state"
        })
        outputs.append({
            "name": "read_file",
            "output": code
        })

        if category in ["code_editing", "bug_fixing", "security_fixes"]:
            # Read + Analyze + Edit workflow
            calls.append({
                "name": "analyze_code",
                "arguments": {"code": code, "focus": "correctness" if category == "bug_fixing" else "style"},
                "thought": "Analyze the code to identify specific issues"
            })
            outputs.append({
                "name": "analyze_code",
                "output": f"Identified issue in {file_path}: needs modification"
            })

            calls.append({
                "name": "edit_file",
                "arguments": {
                    "path": file_path,
                    "old_string": code[:min(60, len(code))],
                    "new_string": self._transform_code(code, category)
                },
                "thought": "Apply the targeted fix"
            })
            outputs.append({
                "name": "edit_file",
                "output": f"Successfully modified {file_path}"
            })

        elif category in ["feature_implementation", "testing"]:
            # Read + Write/Edit workflow
            calls.append({
                "name": "edit_file",
                "arguments": {
                    "path": file_path,
                    "old_string": "# TODO: implement",
                    "new_string": "# TODO: implement\n    pass  # implemented"
                },
                "thought": "Add the new feature/test"
            })
            outputs.append({
                "name": "edit_file",
                "output": f"Added implementation to {file_path}"
            })

        elif category in ["refactoring", "code_optimization"]:
            # Read + Analyze + Edit + Verify workflow
            calls.append({
                "name": "analyze_code",
                "arguments": {"code": code, "focus": "performance" if category == "code_optimization" else "structure"},
                "thought": "Analyze code structure for refactoring opportunities"
            })
            outputs.append({
                "name": "analyze_code",
                "output": "Found opportunities for improvement"
            })

            calls.append({
                "name": "edit_file",
                "arguments": {
                    "path": file_path,
                    "old_string": code[:min(50, len(code))],
                    "new_string": self._transform_code(code, category)
                },
                "thought": "Apply refactoring/optimization"
            })
            outputs.append({
                "name": "edit_file",
                "output": f"Refactored {file_path}"
            })

            calls.append({
                "name": "run_code",
                "arguments": {"language": "python" if "python" in file_path else "shell", "code": f"python -c 'import ast; ast.parse(open(\"{file_path}\").read())'"},
                "thought": "Validate the code is syntactically correct"
            })
            outputs.append({
                "name": "run_code",
                "output": "Syntax valid"
            })

        return calls, outputs

    def _transform_code(self, code: str, category: str) -> str:
        """Transform code based on category."""
        if "typo_bug" in code:
            return code.replace("proces_", "process_")
        if "hardcoded_secret" in code.lower() or "API_KEY" in code:
            return "# API_KEY from env\nAPI_KEY = os.getenv('API_KEY')"
        if "np.int" in code:
            return code.replace("np.int", "np.int64")
        if "86400000" in code:
            return code.replace("86400000", "MS_PER_DAY = 86400000")
        if "var " in code:
            return code.replace("var ", "const ")
        if "string concatenation" in code.lower() or "'" in code and "+" in code:
            return code.replace("'Hello ' + name + '!'", "f'Hello {name}!'")
        return "# refactored: " + code[:min(60, len(code))]

    def generate(
        self,
        category: Optional[str] = None,
        language: Optional[str] = None,
        instruction: Optional[str] = None
    ) -> GoldenSample:
        """Generate a single golden sample."""

        # Select category and task
        if category and instruction:
            task = instruction
        else:
            category, task = self._select_task()

        language = language or self.rng.choice(LANGUAGES)
        file_path = self._get_file(language)
        code = self._get_code(language, category)
        flow = self._get_flow(category)

        # Generate tool chain
        tool_calls, tool_outputs = self._generate_tool_chain(category, file_path, code, task)

        # Get reasoning steps
        reasoning_steps = REASONING_FLOWS.get(flow, REASONING_FLOWS["plan_execute_validate_fix"])

        context = {
            "project": "cli_tool" if self.rng.random() > 0.3 else "web_service" if self.rng.random() > 0.5 else "data_pipeline",
            "language": language,
            "task_type": category,
            "difficulty": self.rng.choice(["easy", "medium", "medium", "hard"]),
            "file_path": file_path,
        }

        final_output = {
            "status": "success",
            "response": f"Completed: {task[:120]}",
            "code": None,
            "explanation": f"Successfully used {len(tool_calls)} tool calls to complete {category}",
            "tool_usage": [
                {"tool": tc["name"], "purpose": "context" if i == 0 else "modification"}
                for i, tc in enumerate(tool_calls)
            ],
            "next_actions": []
        }

        # Quality tags for filtering
        quality_tags = []
        if len(tool_calls) >= 3:
            quality_tags.append("multi_step")
        if category in ["bug_fixing", "security_fixes"]:
            quality_tags.append("high_stakes")
        if "refactor" in category or "optim" in category:
            quality_tags.append("complex_task")

        return GoldenSample(
            id=self._uid(),
            timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S") + "Z",
            schema_version="1.0",
            localization={"language": "en", "tone": "professional", "style": "technical"},
            rules=RULES,
            guardrails=GUARDRAILS,
            available_tools=STANDARD_TOOLS,
            instruction=task,
            context=context,
            reasoning=reasoning_steps[:4],
            reasoning_flow=flow,
            tool_calls=tool_calls,
            tool_outputs=tool_outputs,
            final_output=final_output,
            quality_tags=quality_tags
        )

    def generate_batch(self, count: int) -> list[GoldenSample]:
        """Generate multiple samples with distribution matching weights."""
        samples = []
        for _ in range(count):
            samples.append(self.generate())
        return samples


# =============================================================================
# VALIDATOR
# =============================================================================

class GoldenValidator:
    """Strict validation for golden dataset quality."""

    REQUIRED_FIELDS = ["id", "instruction", "context", "tool_calls", "final_output",
                       "available_tools", "reasoning", "rules", "guardrails"]

    @classmethod
    def validate(cls, sample: dict) -> tuple[bool, list[str]]:
        errors = []

        for field in cls.REQUIRED_FIELDS:
            if field not in sample:
                errors.append(f"Missing field: {field}")

        if len(sample.get("available_tools", [])) > 13:
            errors.append(f"Too many tools: {len(sample['available_tools'])} > 13")

        valid_tools = {t["name"] for t in sample.get("available_tools", [])}
        for tc in sample.get("tool_calls", []):
            if tc.get("name") not in valid_tools:
                errors.append(f"Unknown tool: {tc['name']}")

        try:
            json.loads(json.dumps(sample["instruction"]))
        except Exception:
            errors.append("Instruction not JSON serializable")

        if len(sample.get("reasoning", [])) < 2:
            errors.append("Reasoning chain too short")

        return len(errors) == 0, errors

    @classmethod
    def validate_file(cls, filepath: str) -> dict:
        total = valid = 0
        errors = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    sample = json.loads(line)
                    is_valid, errs = cls.validate(sample)
                    if is_valid:
                        valid += 1
                    else:
                        errors.append({"line": line_num, "errors": errs})
                except json.JSONDecodeError as e:
                    errors.append({"line": line_num, "errors": [f"JSON: {e}"]})

        return {"total": total, "valid": valid, "errors": errors}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--output", type=str, default="golden_full.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", type=str, help="Validate existing file")
    args = parser.parse_args()

    if args.validate:
        result = GoldenValidator.validate_file(args.validate)
        print(f"Total: {result['total']}, Valid: {result['valid']}, "
              f"Invalid: {result['total'] - result['valid']}")
        if result['errors']:
            print(f"\nFirst 10 errors:")
            for e in result['errors'][:10]:
                print(f"  Line {e['line']}: {e['errors']}")
        exit(0)

    print(f"Generating {args.count} golden samples...")
    gen = GoldenDatasetGenerator(seed=args.seed)
    samples = gen.generate_batch(args.count)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.to_jsonl() + "\n")

    size_mb = output_path.stat().st_size / (1024*1024)
    print(f"Written: {output_path} ({size_mb:.2f} MB)")

    result = GoldenValidator.validate_file(str(output_path))
    pct = 100 * result["valid"] / result["total"] if result["total"] else 0
    print(f"Validation: {result['valid']}/{result['total']} ({pct:.1f}%)")

    if result["errors"]:
        print(f"  Errors: {len(result['errors'])}")