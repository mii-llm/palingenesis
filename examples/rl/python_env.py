"""A Python tool for RL: the policy runs code in the sandbox and reads its output.

Any class works as an environment: its public methods are the tools (schemas from the
type hints and docstring), reset(**row) starts an episode, get_reward() is optional.
"""

from palingenesis.rl.sandbox import ExecJob


class PythonEnv:
    def __init__(self, sandbox):
        self.sandbox = sandbox  # the trainer's sandbox (configured in sandbox:)

    def reset(self, **row):
        self.calls = 0

    async def python(self, code: str) -> str:
        """Run a Python program and return what it prints (stdout, then stderr).

        Args:
            code: The complete program. Print the values you need to see.
        """
        self.calls += 1
        (result,) = await self.sandbox.run([ExecJob({"main.py": code}, timeout=10)])
        if result.status == "timeout":
            return "TimeoutError: the program ran longer than 10 s."
        output = result.stdout + (f"\nstderr:\n{result.stderr}" if result.stderr.strip() else "")
        return output.strip() or "(no output)"
