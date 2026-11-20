"""lead_workers: a lead splits a task, workers answer the parts, the lead merges; scored by the run's rewards.

Three stages on any single-agent RL row:

  plan     the lead is given the task and writes at most `max_workers` lines `SUBTASK: ...`. It is told not to
           solve the task yet, and that a worker sees nothing but its own subtask.
  work     each subtask goes to one worker session, which sees that subtask alone (cut to `max_subtask_chars`).
  merge    the lead's conversation continues: after its own plan it is given the subtasks with the workers'
           replies (each cut to `max_reply_chars`) and gives the final answer, which the rewards score.

The work and merge stages run `team_samples` times from the same plan (a branch), and each team's merge
`merge_samples` times from the same replies (a branch inside it). One more session, the control, is the lead
answering the task alone, with the prompt a single agent would get.

Roles: `lead`, `worker`.

Credit, by the rules of the guide (a baseline only among sessions that could have been each other):

  lead, plan     total: the mean reward of every merge that followed the plan. Compared with the plans of the
                 row's other episodes, which started from the same prompt.
                 lift: total minus the control's reward: what delegating added. Weight 0 by default; make it
                 the plan's reward with weights: {lead: {total: 0, lift: 1}}.
                 metric/subtasks, metric/control are logged.
  lead, merge    correct: the rewards' weighted sum on its final answer. Compared with the other merges of the
                 same team, which read the same replies: a merge is not credited with its workers' luck.
  worker         team: the mean reward of its team's merges. Compared with the workers of the other teams that
                 answered the same subtask (the subtask's index is the session's slot): sessions that received
                 different subtasks are never compared.

Reward hacking, by policy:

  a lead that solves alone      at the merge and ignores the replies is paid in full: `total` pays the final
                                answer, not the delegation. Its workers' rewards then carry no signal (equal
                                within each comparison, so dropped as zero variance), and `lift` is 0. This is
                                the single-agent control the guide asks for, measured inside the run: when
                                metric/control is close to the reward, the pattern adds tokens and nothing else.
  a plan with no subtask        is allowed (no worker runs, the merge answers alone): same as above.
  the whole task as a subtask   moves the task to a worker. It is a valid delegation; the split is not judged.
  an idle or wrong worker       shares its team's reward: one worker's contribution is separated from its
                                teammates' only on average over teams, so worker credit is noisy with few
                                teams. A sub-answer cannot be checked by itself here (the dataset has gold for
                                the task, not for its parts); where it can, write a scenario that does, as
                                ledger_sum does.
  workers that never matter     earn the same in every team and are dropped (zero variance): harmless, and a
                                sign that the lead does not use them.
  lenient rewards               pay the control as much as the team: lift 0 everywhere.

The oracle partners delegate for real: the oracle lead hands the whole task to each worker, gives the right answer
at the merge only when a worker's reply holds it, and answers nothing alone; the oracle worker gives the gold
answer. A silent worker therefore earns nothing in the audit.
"""

from __future__ import annotations

import asyncio
import re

from palingenesis.rl.scenarios.base import assistant, user
from palingenesis.rl.scenarios.library.patterns import Pattern, mean
from palingenesis.rl.scenarios.runtime import Episode

PLAN = {
    "en": (
        "{task}\n\nYou lead a team of {n} assistants. They cannot see the task above: each one sees only the "
        "subtask you write for them. Do not solve the task yet. Split it into at most {n} subtasks, one per line, "
        "each of the form:\nSUBTASK: <a question or an instruction that can be understood alone>"
    ),
    "it": (
        "{task}\n\nGuidi un gruppo di {n} assistenti. Non possono vedere il compito qui sopra: ognuno vede solo "
        "il sottocompito che scrivi per lui. Non risolvere ancora il compito. Dividilo in al massimo {n} "
        "sottocompiti, uno per riga, ciascuno nella forma:\nSUBTASK: <una domanda o un'istruzione comprensibile da sola>"
    ),
}
WORKER = {
    "en": (
        "You are helping a colleague with one part of a larger task that you cannot see. Your part:\n\n{subtask}\n\n"
        "Reply briefly, with your result."
    ),
    "it": (
        "Stai aiutando un collega con una parte di un compito più grande che non puoi vedere. La tua parte:\n\n"
        "{subtask}\n\nRispondi in breve, con il tuo risultato."
    ),
}
MERGE = {
    "en": "Your assistants' replies:\n\n{replies}\n\nNow give the final answer to the task.",
    "it": "Le risposte dei tuoi assistenti:\n\n{replies}\n\nOra dai la risposta finale al compito.",
}
BLOCK = {"en": "Subtask {i}: {subtask}\nReply {i}: {reply}", "it": "Sottocompito {i}: {subtask}\nRisposta {i}: {reply}"}
NOTHING = {
    "en": "(no subtask was given, so there are no replies)",
    "it": "(nessun sottocompito, quindi nessuna risposta)",
}
SUBTASK = re.compile(r"(?im)^[ \t>*_`\-\d.)]*SUBTASK\b[ \t*_`]*:[ \t]*(.+?)[ \t]*$")


class LeadWorkers(Pattern):
    name = "lead_workers"
    roles = ("lead", "worker")
    outcome = "lead"
    weights = {"lead": {"total": 1.0, "lift": 0.0, "correct": 1.0}, "worker": {"team": 1.0}}
    # the oracle lead's sessions are its plans (paid `total`) and its merges (paid `correct`): 1 each
    oracle_floor = {"lead": 0.9, "worker": 0.9}

    def __init__(
        self,
        weights=None,
        max_workers: int = 3,
        team_samples: int = 2,
        merge_samples: int = 2,
        max_subtask_chars: int = 600,
        max_reply_chars: int = 400,
        control: bool = True,
        **options,
    ):
        super().__init__(weights, **options)
        if min(max_workers, team_samples, merge_samples) < 1:
            raise ValueError("lead_workers: max_workers, team_samples and merge_samples must be >= 1")
        self.max_workers, self.team_samples, self.merge_samples = max_workers, team_samples, merge_samples
        self.max_subtask_chars, self.max_reply_chars, self.control = max_subtask_chars, max_reply_chars, control

    # ------------------------------------------------------------------ prompts

    def plan_messages(self, row: dict) -> list[dict]:
        messages = self.conversation(row)
        text = PLAN[self.language(row)].format(task=messages[-1]["content"].strip(), n=self.max_workers)
        return [*messages[:-1], user(text)]

    def merge_messages(self, row: dict, plan: str, subtasks: list[str], replies: list[str]) -> list[dict]:
        language = self.language(row)
        blocks = [
            BLOCK[language].format(i=i + 1, subtask=subtask, reply=reply.strip())
            for i, (subtask, reply) in enumerate(zip(subtasks, replies))
        ]
        text = MERGE[language].format(replies="\n\n".join(blocks) or NOTHING[language])
        return [*self.plan_messages(row), assistant(plan), user(self.with_answer_line(text, language))]

    def subtasks(self, plan: str) -> list[str]:
        """The plan's `SUBTASK:` lines, in order (all of them: the caller keeps the first max_workers)."""
        return [found.strip().strip("*_`").strip() for found in SUBTASK.findall(plan or "") if found.strip()]

    # ------------------------------------------------------------------ partners

    def partners(self):
        def stage(messages, row) -> str:
            text = messages[-1]["content"]
            if text.startswith(MERGE[self.language(row)].split("{replies}")[0]):
                return "merge"
            return "plan" if text == self.plan_messages(row)[-1]["content"] else "control"

        def flat_task(row) -> str:
            return " ".join(self.task(row).split())

        def oracle_lead(messages, row):
            now, gold = stage(messages, row), self.gold(row)
            if now == "plan":  # the whole task to every worker
                return "\n".join(f"SUBTASK: {flat_task(row)}" for _ in range(self.max_workers))
            if now == "merge" and gold is not None:  # the answer only if a worker's reply holds it
                replies = messages[-1]["content"].replace(flat_task(row)[: self.max_subtask_chars], "")
                replies = replies.replace(self.answer_line(self.language(row)), "")
                return self.say(gold) if gold.casefold() in replies.casefold() else ""
            return ""  # the control: alone, it does not know

        def no_merge(messages, row):
            return oracle_lead(messages, row) if stage(messages, row) == "plan" else ""

        def oracle_worker(messages, row):
            return self.gold(row) or ""

        return {
            "lead": {"oracle": oracle_lead, "silent": lambda messages: "", "no_merge": no_merge},
            "worker": {
                "oracle": oracle_worker,
                "silent": lambda messages: "",
                "unsure": lambda messages: "I am not sure.",
            },
        }

    # ------------------------------------------------------------------ episode

    async def run(self, episode: Episode, row: dict) -> None:
        self.required(row)
        language = self.language(row)
        plan = await episode.run("lead", self.plan_messages(row), slot="plan")
        text = plan.text if plan.ok else ""
        found = self.subtasks(text)
        handed = episode.channel("subtask", max_chars=self.max_subtask_chars)
        returned = episode.channel("reply", max_chars=self.max_reply_chars)
        subtasks = [handed.send(s, "lead", f"worker:{i}") for i, s in enumerate(found[: self.max_workers])]
        plan.reward("metric/subtasks", len(found))

        async def team(branch: Episode) -> float | None:
            workers = await asyncio.gather(
                *(
                    branch.run("worker", [user(WORKER[language].format(subtask=subtask))], slot=str(i))
                    for i, subtask in enumerate(subtasks)
                )
            )
            replies = [returned.send(w.text if w.ok else "", f"worker:{i}", "lead") for i, w in enumerate(workers)]
            prompt = self.merge_messages(row, text, subtasks, replies)

            async def merge(leaf: Episode) -> float | None:
                session = await leaf.run("lead", prompt, slot="merge")
                return (await self.pay(session, row)).score

            score = mean(await branch.branch(self.merge_samples, merge, label="merge"))
            for worker in workers:
                worker.reward("team", score)
            return score

        async def alone() -> float | None:
            control = await episode.run("lead", self.solver_messages(row), slot="control")
            score = (await self.grade(self.answer_of(control), row, control)).score
            control.reward("metric/control", score)  # logged only: the control's session does not train
            return score

        teams = episode.branch(self.team_samples, team, label="team")
        if not self.control:
            plan.reward("total", mean(await teams))
            return
        scores, baseline = await asyncio.gather(teams, alone())
        total = mean(scores)
        plan.reward("total", total)
        plan.reward("metric/control", baseline)
        plan.reward("lift", None if total is None or baseline is None else total - baseline)
