"""plan_execute: a planner who sees a Harbor task and an executor who carries it out in the task's container.

Rows are Harbor rows as `pgs harbor rows` writes them: {"task_dir", "messages": [system, instruction]}. The
planner is given the task's instruction and no tools, and writes a plan of at most `max_plan_chars` characters.
The executor gets the row's own conversation with the plan appended, and the tools of the environment: a fresh
HarborEnv per executor session (a container, bash / read_file / write_file / str_replace / submit). When the
executor submits or stops, the task's tests give the reward, exactly as in single-agent RL on Harbor.

The environment is used as it is (`env`, default palingenesis.rl.envs.harbor:HarborEnv, with `env_args`): reset
with the row, its tool schemas and tool calls behind the session's tools, get_reward on the transcript, close.
Any environment class with tool_schemas / call_tool / get_reward works the same way.

`executor_samples` executors start from the same plan, each in its own container (a branch: one comparison
group). With `control: true` one more executor works without a plan.

At most `max_concurrent` environments are alive at once, per trainer process (per rank under torchrun). An
executor waiting for a free one is inside its episode, so the wait counts against `scenario.timeout`: keep
max_concurrent at or above rollout.batch_prompts x rollout.group_size x the executors per episode, or raise the
timeout. An episode that times out is left out of training, planner included.

Roles: `planner`, `executor`.

Reward components:

  planner    executed        the mean reward of its executors. The planner's default reward.
             lift            `executed` minus the control's reward (needs `control: true`): what the plan added.
                             Weight 0 by default.
             metric/plan_chars, metric/plan_cut
  executor   tests           the environment's reward (the task's tests); the environment's metric/... are logged.
                             An executor whose verifier failed is left out of the loss.
             (the control's session is logged as metric/control and is not trained)

Reward hacking, by policy:

  the planner               has no tools and never touches the container: it can only change what the executor
                            reads. A plan that contains the commands to run is a plan.
  an executor that ignores  is paid by the tests all the same, and so is its planner: `executed` pays the
  the plan                  planner for tasks the executor solves alone. Turn `control` on and train the planner
                            on `lift` (weights: {planner: {executed: 0, lift: 1}}) when that matters; it costs
                            one more container per episode.
  an empty plan             earns what the control earns: lift 0. A constant plan of generic advice ("read the
                            task twice, run the tests") can earn a lift without planning anything: the
                            control has no plan at all, so the lift also pays what any extra text adds.
  the tests                 are the task's own, with the single-agent holes they have (run `pgs harbor check`:
                            the oracle must pass, doing nothing must fail). This pattern adds none: the plan is
                            text, cut to `max_plan_chars`, and the tests never see it.
  a private code            between a planner and an executor trained together is possible, as in handoff:
                            evaluate with an untrained executor.

The oracle partners need the task's reference solution: the oracle planner writes solution/solve.sh as its plan,
and the oracle executor runs the script it finds in the plan and submits. A silent planner therefore leaves the
oracle executor with nothing to run. Tasks graded on the final message (solution/final_message.md) have no oracle
here.

Limits: harness profiles (HarborEnv's profile / profiles) are refused, because a scenario session's conversation
and text protocol are the scenario's own. The two roles do not share a container. Tested with a stand-in
environment; not yet run against Docker.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from pathlib import Path
from typing import Any

from palingenesis.rl.scenarios.base import LANGUAGES, Scenario, user
from palingenesis.rl.scenarios.library.patterns import mean
from palingenesis.rl.scenarios.runtime import EndSession, Episode, ScenarioError

PLANNER = {
    "en": (
        "A colleague will carry out the task below in a Linux container, with a shell and file tools. You cannot "
        "run anything yourself. Write a plan for them, in at most {limit} characters: the steps to take, what to "
        "check, and the mistakes to avoid.\n\nTask:\n{task}"
    ),
    "it": (
        "Un collega svolgerà il compito qui sotto in un container Linux, con una shell e strumenti per i file. Tu "
        "non puoi eseguire nulla. Scrivi per lui un piano, in al massimo {limit} caratteri: i passi da fare, cosa "
        "verificare e gli errori da evitare.\n\nCompito:\n{task}"
    ),
}
PLAN = {
    "en": "\n\nA colleague's plan for this task:\n{plan}",
    "it": "\n\nIl piano di un collega per questo compito:\n{plan}",
}
SCRIPT = re.compile(r"```bash\n(.*?)\n```", re.S)
logger = logging.getLogger(__name__)


class PlanExecute(Scenario):
    name = "plan_execute"
    roles = ("planner", "executor")
    outcome = "executor"
    weights = {"planner": {"executed": 1.0, "lift": 0.0}, "executor": {"tests": 1.0}}
    check_rows = 5  # rows `pgs scenario check` plays by default: every executor session is a container
    fields = {
        "task_dir": "the Harbor task directory (what the environment's reset takes)",
        "messages": "the executor's conversation: [system,] the task's instruction (`pgs harbor rows` writes it)",
        "max_turns": "optional: the executor's turn limit for this row",
        "language": "optional: en or it (scenario.args.language_field)",
    }

    def __init__(
        self,
        weights=None,
        env: str = "palingenesis.rl.envs.harbor:HarborEnv",
        env_args: dict | None = None,
        executor_samples: int = 2,
        max_plan_chars: int = 4000,
        max_turns: int = 16,
        tool_timeout: float = 120.0,
        max_concurrent: int = 8,
        control: bool = False,
        prompt_field: str = "",
        language_field: str = "language",
        language: str = "en",
    ):
        super().__init__(weights)
        self.env_args = dict(env_args or {})
        if self.env_args.get("profile") or self.env_args.get("profiles"):
            raise ValueError("plan_execute: harness profiles (env_args.profile / profiles) are not supported")
        if language not in LANGUAGES:
            raise ValueError(f"plan_execute: language must be one of {LANGUAGES}, got {language!r}")
        if min(executor_samples, max_plan_chars, max_turns, max_concurrent) < 1 or not tool_timeout > 0:
            raise ValueError(
                "plan_execute: executor_samples, max_plan_chars, max_turns, max_concurrent and tool_timeout must be > 0"
            )
        self.env, self.executor_samples, self.max_plan_chars = env, executor_samples, max_plan_chars
        self.max_turns, self.tool_timeout, self.max_concurrent = max_turns, float(tool_timeout), max_concurrent
        self.control, self.prompt_field = control, prompt_field
        self.language_field, self.default_language = language_field, language
        self._factory: Any = None
        self._slots: dict[Any, asyncio.Semaphore] = {}
        self._closing: set = set()  # shutdowns in flight (kept referenced until they end)

    @classmethod
    def generate(cls, n: int, seed: int = 0, **params) -> list[dict]:
        raise NotImplementedError(
            "plan_execute is a pattern: it has no generator. Its rows are Harbor rows (`pgs harbor rows`); validate "
            "them with `pgs scenario check --config <rl.yaml>`."
        )

    def bind(self, rewards: list, sandbox: Any = None, config: Any = None) -> None:
        if config is not None:
            self.prompt_field = self.prompt_field or config.data.prompt_field

    # ------------------------------------------------------------------ the row

    def language(self, row: dict) -> str:
        value = str(row.get(self.language_field) or "").strip().lower()[:2]
        return value if value in LANGUAGES else self.default_language

    def conversation(self, row: dict) -> list[dict]:
        from palingenesis.rl.data import prompt_messages

        where = f"plan_execute: row {str(row.get('task_dir', row.get('id', '')))[-60:]!r}"
        if not row.get("task_dir"):
            raise ScenarioError(f"{where}: no 'task_dir' (the Harbor task directory). Columns: {sorted(row)}.")
        try:
            messages = prompt_messages(row, self.prompt_field)
        except (KeyError, ValueError, TypeError) as error:
            raise ScenarioError(
                f"{where}: no prompt ({error}). Write the rows with `pgs harbor rows`, or name the column with "
                "scenario.args.prompt_field."
            ) from None
        last = messages[-1]
        if last.get("role") != "user" or not isinstance(last.get("content"), str) or not last["content"].strip():
            raise ScenarioError(f"{where}: the prompt must end with a user message of text (the task's instruction).")
        return messages

    def check_row(self, row: dict) -> list[str]:
        try:
            self.conversation(row)
        except ScenarioError as error:
            return [str(error)]
        return []

    def planner_prompt(self, row: dict) -> str:
        task = self.conversation(row)[-1]["content"].strip()
        return PLANNER[self.language(row)].format(task=task, limit=self.max_plan_chars)

    def executor_messages(self, row: dict, plan: str) -> list[dict]:
        messages = self.conversation(row)
        if not plan.strip():
            return messages
        text = messages[-1]["content"].rstrip() + PLAN[self.language(row)].format(plan=plan.strip())
        return [*messages[:-1], user(text)]

    # ------------------------------------------------------------------ partners

    def partners(self):
        def oracle_planner(messages, row):
            script = Path(str(row.get("task_dir", ""))) / "solution" / "solve.sh"
            if not script.is_file():
                return "Follow the task's instruction."
            return f"Run this script:\n```bash\n{script.read_text(encoding='utf-8').strip()}\n```"

        async def call(tools, name, **arguments):
            try:
                return await tools[name](**arguments) if name in tools else None
            except EndSession as end:
                return str(end)
            except Exception as error:  # noqa: BLE001 — a failed tool call is an observation, for a partner too
                return f"Error: {error}"

        async def oracle_executor(messages, tools):  # it runs the script of the plan it was given, if any
            task = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
            plan = ""
            for template in PLAN.values():
                head, found, rest = task.rpartition(template.split("{plan}")[0])
                plan = rest if found else plan
            scripts = SCRIPT.findall(plan)
            if scripts:
                await call(tools, "bash", command=scripts[-1])
            await call(tools, "submit")
            return "Done."

        async def idle(messages, tools):
            await call(tools, "submit")
            return "Done."

        return {
            "planner": {
                "oracle": oracle_planner,
                "silent": lambda messages: "",
                "restate": lambda messages: messages[-1]["content"].split("\n\n")[-1],
            },
            "executor": {"oracle": oracle_executor, "idle": idle, "silent": lambda messages: ""},
        }

    # ------------------------------------------------------------------ the environment

    def _environment(self) -> Any:
        if self._factory is None:
            from palingenesis.rl.rewards import load_object

            self._factory = load_object(self.env)
        return self._factory(**self.env_args)

    def _slot(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if loop not in self._slots:  # a semaphore belongs to the event loop it is first used on
            self._slots = {loop: asyncio.Semaphore(self.max_concurrent)}
        return self._slots[loop]

    @staticmethod
    async def _shutdown(env: Any, reset: Any, slot_lock: asyncio.Semaphore) -> None:
        from palingenesis.rl.env import call_sync_or_async

        try:
            if reset is not None:
                await asyncio.gather(reset, return_exceptions=True)
            closer = getattr(env, "aclose", None) or getattr(env, "close", None)
            if closer is not None:
                await call_sync_or_async(closer)
        except Exception:  # noqa: BLE001 — nothing can be done about a close that fails, except saying so
            logger.exception("plan_execute: closing the environment failed")
        finally:
            slot_lock.release()

    async def execute(self, episode: Episode, row: dict, plan: str, slot: str = "") -> tuple[Any, float | None]:
        """One executor session in an environment of its own: (session, the tests' reward or None)."""
        from palingenesis.rl.data import row_limit
        from palingenesis.rl.env import ToolFailed, call_sync_or_async, run_tool

        slot_lock = self._slot()
        await slot_lock.acquire()  # (waiting here counts against scenario.timeout: see max_concurrent)
        env, reset = None, None
        try:
            env = self._environment()
            # shielded: a reset that an episode's cancellation interrupted would start its container afterwards
            reset = asyncio.ensure_future(call_sync_or_async(env.reset, **row))
            opening = await asyncio.shield(reset)
            if opening is not None:
                raise ScenarioError(
                    "plan_execute: the environment's reset returned a conversation of its own (a harness "
                    "profile?); the executor's conversation is the row's, with the plan."
                )

            def tool(name: str, schema: dict):
                async def call(**arguments):
                    result, failed = await run_tool(env, name, arguments, self.tool_timeout)
                    if getattr(env, "done", False):  # a submit: the session takes no further turn
                        raise EndSession(result if isinstance(result, str) else str(result))
                    if failed:  # counted as a tool error, and worded "Error: ..." again by the turn loop
                        raise ToolFailed(str(result).removeprefix("Error: "))
                    return result

                call.schema = schema
                return call

            schemas = await call_sync_or_async(env.tool_schemas)
            tools = {schema["function"]["name"]: tool(schema["function"]["name"], schema) for schema in schemas}
            limits = {"max_turns": row_limit(row, "max_turns") or self.max_turns}
            if row_limit(row, "max_completion_tokens"):
                limits["max_completion_tokens"] = row_limit(row, "max_completion_tokens")
            session = await episode.run("executor", self.executor_messages(row, plan), tools=tools, slot=slot, **limits)
            try:  # as in single-agent RL: an environment that cannot score leaves the sample out, nothing more
                wants_messages = bool(inspect.signature(env.get_reward).parameters)
                value = await call_sync_or_async(env.get_reward, *((session.messages,) if wants_messages else ()))
            except Exception:  # noqa: BLE001
                logger.exception("plan_execute: the environment's get_reward failed; the executor is not scored")
                value = None
        finally:
            # the environment is closed after its reset has finished, and its slot is freed after the close, also
            # when the episode was cancelled (the shutdown then goes on in the background)
            shutdown = asyncio.ensure_future(self._shutdown(env, reset, slot_lock))
            self._closing.add(shutdown)
            shutdown.add_done_callback(self._closing.discard)
            await asyncio.shield(shutdown)
        components = value if isinstance(value, dict) else {"reward": value}
        failed = bool(components.get("metric/verifier_failed"))
        paid = [float(v) for k, v in components.items() if v is not None and not k.startswith("metric/")]
        reward = None if failed or not paid else sum(paid)
        for name, metric in components.items():
            if name.startswith("metric/") and metric is not None:
                session.reward(name, float(metric))
        return session, reward

    # ------------------------------------------------------------------ episode

    async def run(self, episode: Episode, row: dict) -> None:
        planner = await episode.run("planner", [user(self.planner_prompt(row))])
        channel = episode.channel("plan", max_chars=self.max_plan_chars)
        plan = channel.send(planner.text if planner.ok else "", "planner", "executor")
        planner.reward("metric/plan_chars", len(plan))
        planner.reward("metric/plan_cut", episode.log[-1]["cut"])

        async def carry_out(branch: Episode) -> float | None:
            session, reward = await self.execute(branch, row, plan)
            session.reward("tests", reward)
            return reward

        async def unplanned() -> float | None:
            session, reward = await self.execute(episode, row, "", slot="control")
            session.reward("metric/control", reward)  # logged only: the control's session does not train
            return reward

        executors = episode.branch(self.executor_samples, carry_out, label="execute")
        if not self.control:
            planner.reward("executed", mean(await executors))
            return
        rewards, baseline = await asyncio.gather(executors, unplanned())
        executed = mean(rewards)
        planner.reward("executed", executed)
        planner.reward("metric/control", baseline)
        planner.reward("lift", None if executed is None or baseline is None else executed - baseline)
