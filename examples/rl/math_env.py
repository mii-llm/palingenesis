"""Agentic math: the policy computes with a Python tool and commits its result with a submit
tool, which ends the episode; the environment grades the submitted answer.

Every program runs in the trainer's sandbox (configured in sandbox:), never on the host.
"""

from palingenesis.rl.grading import math_equal
from palingenesis.rl.sandbox import ExecJob


class MathToolsEnv:
    tools = ("python", "submit_answer")  # the policy's tools; nothing else here is callable by it

    def __init__(self, sandbox, timeout: float = 10.0):
        self.sandbox = sandbox  # the trainer's sandbox
        self.timeout = timeout

    def reset(self, answer: str, **row):
        self.expected, self.submitted, self.done = str(answer), None, False

    async def python(self, code: str) -> str:
        """Run a Python 3 program and return what it prints (stdout, then stderr).

        Only the standard library is available (math, fractions, decimal, itertools, ...), with
        no network or files from earlier calls: each call is a fresh program, so print every
        value you need.

        Args:
            code: The complete program.
        """
        (result,) = await self.sandbox.run([ExecJob({"main.py": code}, timeout=self.timeout)])
        if result.status == "timeout":
            return f"TimeoutError: the program ran longer than {self.timeout:g} s."
        output = result.stdout + (f"\nstderr:\n{result.stderr}" if result.stderr.strip() else "")
        return output.strip() or "(no output: print the values you need)"

    def submit_answer(self, answer: str) -> str:
        """Submit the final answer and end the task. Call it exactly once, when you are sure.

        Args:
            answer: The final answer only: a number or a simplified expression, without units,
                words or \\boxed{}.
        """
        self.submitted, self.done = answer, True
        return "Answer submitted."

    def get_reward(self) -> float:
        return float(math_equal(self.submitted, self.expected))
