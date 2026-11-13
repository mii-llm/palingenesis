"""Tests for data validation: role normalization and tool-call consistency."""

import sys

sys.path.insert(0, "src")


def test_role_normalization_sharegpt():
    """ShareGPT format (from/value, human/gpt) normalizes correctly."""
    from palingenesis.validate_data import normalize_messages

    sample = {
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "4"},
        ]
    }

    result = normalize_messages(sample, "conversations")
    assert result is not None
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "What is 2+2?"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "4"
    print("✓ test_role_normalization_sharegpt PASSED\n")


def test_role_normalization_standard():
    """Standard format passes through unchanged."""
    from palingenesis.validate_data import normalize_messages

    sample = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
    }

    result = normalize_messages(sample, "messages")
    assert result is not None
    assert len(result) == 3
    assert result[0]["role"] == "system"
    assert result[2]["role"] == "assistant"
    assert result[2]["content"] == "Hello!"
    print("✓ test_role_normalization_standard PASSED\n")


def test_role_normalization_with_reasoning():
    """Reasoning content and tool_calls are preserved during normalization."""
    from palingenesis.validate_data import normalize_messages

    sample = {
        "messages": [
            {"role": "user", "content": "Fix the bug"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Let me think...",
                "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "src/main.py"}'}}],
            },
            {"role": "tool", "content": "def main(): pass"},
        ]
    }

    result = normalize_messages(sample, "messages")
    assert result[1]["reasoning_content"] == "Let me think..."
    assert result[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert result[2]["role"] == "tool"
    print("✓ test_role_normalization_with_reasoning PASSED\n")


def test_role_normalization_think_field():
    """The 'think' field (alternative to reasoning_content) is normalized."""
    from palingenesis.validate_data import normalize_messages

    sample = {
        "messages": [
            {"role": "user", "content": "Solve x^2=4"},
            {"role": "assistant", "content": "x=2 or x=-2", "think": "I need to find square roots of 4..."},
        ]
    }

    result = normalize_messages(sample, "messages")
    assert result[1]["reasoning_content"] == "I need to find square roots of 4..."
    print("✓ test_role_normalization_think_field PASSED\n")


def test_role_normalization_alternative_field_names():
    """Finds messages in 'conversations', 'chat', 'dialogue' etc."""
    from palingenesis.validate_data import normalize_messages

    sample = {"dialogue": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello"}]}

    # With wrong field name, tries alternatives
    result = normalize_messages(sample, "messages")
    assert result is not None
    assert result[0]["role"] == "user"
    print("✓ test_role_normalization_alternative_field_names PASSED\n")


def test_tool_validation_valid():
    """Valid tool calls pass validation."""
    from palingenesis.validate_data import validate_tool_calls

    sample = {
        "messages": [
            {"role": "user", "content": "Read the file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "/src/main.py"}'}}],
            },
            {"role": "tool", "content": "file contents here"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                },
            }
        ],
    }

    errors = validate_tool_calls(sample)
    assert errors == [], f"Expected no errors, got: {errors}"
    print("✓ test_tool_validation_valid PASSED\n")


def test_tool_validation_unknown_tool():
    """Calling a tool not in the tools list produces an error."""
    from palingenesis.validate_data import validate_tool_calls

    sample = {
        "messages": [
            {"role": "user", "content": "Do something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "delete_everything", "arguments": "{}"}}],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ],
    }

    errors = validate_tool_calls(sample)
    assert len(errors) == 1
    assert "delete_everything" in errors[0]
    assert "not found" in errors[0]
    print(f"  Error caught: {errors[0][:80]}")
    print("✓ test_tool_validation_unknown_tool PASSED\n")


def test_tool_validation_invalid_param():
    """Passing a parameter not in the schema produces an error."""
    from palingenesis.validate_data import validate_tool_calls

    sample = {
        "messages": [
            {"role": "user", "content": "Read file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": '{"path": "/a.py", "encoding": "utf-8"}'}}
                ],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                },
            }
        ],
    }

    errors = validate_tool_calls(sample)
    assert len(errors) == 1
    assert "encoding" in errors[0]
    print(f"  Error caught: {errors[0][:80]}")
    print("✓ test_tool_validation_invalid_param PASSED\n")


def test_tool_validation_missing_required():
    """Missing a required parameter produces an error."""
    from palingenesis.validate_data import validate_tool_calls

    sample = {
        "messages": [
            {"role": "user", "content": "Read"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}],
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                },
            }
        ],
    }

    errors = validate_tool_calls(sample)
    assert len(errors) == 1
    assert "required" in errors[0]
    assert "path" in errors[0]
    print(f"  Error caught: {errors[0][:80]}")
    print("✓ test_tool_validation_missing_required PASSED\n")


def test_tool_validation_no_tools_field():
    """Samples without tools field produce no errors (nothing to validate)."""
    from palingenesis.validate_data import validate_tool_calls

    sample = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }

    errors = validate_tool_calls(sample)
    assert errors == []
    print("✓ test_tool_validation_no_tools_field PASSED\n")


def test_dataset_validation_report():
    """Full dataset validation produces a structured report."""
    from palingenesis.validate_data import validate_dataset

    samples = [
        {
            "messages": [
                {"role": "user", "content": "Read file"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
        },
        {
            "messages": [
                {"role": "user", "content": "Delete"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "nuke", "arguments": "{}"}}]},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                    },
                }
            ],
        },
    ]

    report = validate_dataset(samples)
    assert report["total"] == 2
    assert report["valid"] == 1
    assert report["samples_with_tools"] == 2
    assert len(report["errors"]) == 1
    assert "read_file" in report["tools_called"]
    print(f"  Report: {report['valid']}/{report['total']} valid, {len(report['errors'])} errors")
    print("✓ test_dataset_validation_report PASSED\n")


_READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}


def test_tool_validation_tolerates_arrow_null_fields():
    """Arrow fills a struct field that a row lacks with None: `tool_calls` on a plain assistant turn, `properties`
    / `required` on a tool without parameters. None of them is an error."""
    from palingenesis.validate_data import validate_dataset, validate_tool_calls

    sample = {
        "messages": [
            {"role": "user", "content": "hi", "tool_calls": None},
            {"role": "assistant", "content": "hello", "tool_calls": None, "function_call": None},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "now", "arguments": None}}],
            },
        ],
        "tools": [
            _READ_FILE,
            {"type": "function", "function": {"name": "now", "parameters": {"properties": None, "required": None}}},
        ],
    }
    assert validate_tool_calls(sample) == []
    report = validate_dataset([sample])
    assert report["valid"] == 1 and report["tools_called"] == {"now": 1}
    print("✓ test_tool_validation_tolerates_arrow_null_fields PASSED\n")


def test_dataset_validation_counts_only_checked_samples_when_truncated():
    """The scan stops at max_errors: samples it never looked at are not `valid`."""
    from palingenesis.validate_data import validate_dataset

    bad = {
        "messages": [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "nope"}}]}],
        "tools": [_READ_FILE],
    }
    report = validate_dataset([bad] * 10, max_errors=3)
    assert report["truncated"] is True
    assert report["total"] == 10 and report["checked"] == 3 and report["valid"] == 0
    full = validate_dataset([bad] * 2 + [{"messages": [{"role": "user", "content": "x"}]}], max_errors=50)
    assert (full["total"], full["checked"], full["valid"], full["truncated"]) == (3, 3, 1, False)
    print("✓ test_dataset_validation_counts_only_checked_samples_when_truncated PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("DATA VALIDATION TESTS")
    print("=" * 60 + "\n")

    test_role_normalization_sharegpt()
    test_role_normalization_standard()
    test_role_normalization_with_reasoning()
    test_role_normalization_think_field()
    test_role_normalization_alternative_field_names()
    test_tool_validation_valid()
    test_tool_validation_unknown_tool()
    test_tool_validation_invalid_param()
    test_tool_validation_missing_required()
    test_tool_validation_no_tools_field()
    test_dataset_validation_report()
    test_tool_validation_tolerates_arrow_null_fields()
    test_dataset_validation_counts_only_checked_samples_when_truncated()

    print("=" * 60)
    print("ALL DATA VALIDATION TESTS PASSED ✓")
    print("=" * 60)
