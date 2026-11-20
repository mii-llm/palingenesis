"""handoff: a writer who sees the material and a reader who sees only the writer's note.

Rows are question answering over a text: the material (`material_field`, default "context"), the question (the
row's prompt, as `pgs rl` reads it; `question_field` names another column) and whatever the run's rewards read
(a gold `answer`). The writer is given the material and not the question, and writes a note of at most
`max_note_chars` characters. The reader is given the note and the question, never the material, and answers. The
rewards score the reader's final answer.

`reader_samples` readers answer from the same note (a branch: one comparison group). One more reader, the
control, answers the question with no note at all: what it scores is what a reader could do without the writer.

Roles: `writer`, `reader`.

Reward components:

  writer    lift            the mean reward of its readers minus the control's: what the note added. The default
                            reward of the writer.
            readers         the mean reward of its readers. Weight 0 by default; the writer's reward when
                            `control` is off.
            metric/note_chars, metric/note_cut   the note's length as delivered, and whether it was cut.
  reader    correct         the rewards' weighted sum on its final answer; 0 without an answer.
            (the control's session is logged as metric/control and is not trained)

Reward hacking, by policy:

  an empty note               reads like no note: the readers score what the control scores and the lift is 0.
  a constant note             of generic advice ("answer in as few words as possible") can earn a lift without
                              a word about the material, when it makes the reader's answers score better: the
                              control has no note at all, so the lift also pays what any text adds. The reader's
                              prompt already asks for the answer line, which leaves little to coach; compare
                              role/writer/rewards/lift with a run whose writer is `scripted:filler`.
  the material in the prompt  a row whose question column already contains the text (single-agent reading rows
                              often do) gives the reader everything: lift 0 for every writer.
                              `pgs scenario check` warns; point question_field at the bare question.
  a reader that knows the     earns its own `correct`, as it should, and nothing for the writer: the control
  answer without the note     knows it too. Without the control (`control: false`) the writer would be paid for
                              questions the reader can answer alone; keep it on unless the questions cannot be
                              answered without the material.
  copying the material        is the best note whenever the material fits in `max_note_chars`, and teaches
                              nothing. `pgs scenario check` warns when most materials fit; lower the limit.
  the answer in the note      cannot be targeted: the writer does not see the question. A writer that guesses the
                              likely question and answers it is doing the task.
  a private code              a writer and a reader trained together can agree on notes only they can read.
                              The reward cannot see that. Train against a reader that does not learn
                              (`roles.reader.partner: base`, or `train: false`), and evaluate with an untrained
                              reader.
  lenient rewards             a reward that pays an empty answer pays the control as much as the readers: the
                              lift is 0 for every writer and the audit reports the oracle writer as unpaid.
"""

from __future__ import annotations

import asyncio

from palingenesis.rl.scenarios.base import user
from palingenesis.rl.scenarios.library.patterns import Pattern, between, mean
from palingenesis.rl.scenarios.runtime import Episode

WRITER = {
    "en": (
        "{material}\n\nA colleague who cannot see the text above will be asked a question about it. You do not "
        "know the question. Write notes for them, in at most {limit} characters, with everything they could need."
    ),
    "it": (
        "{material}\n\nA un collega che non può vedere il testo qui sopra verrà fatta una domanda al riguardo. Tu "
        "non conosci la domanda. Scrivi degli appunti per lui, in al massimo {limit} caratteri, con tutto ciò che "
        "potrebbe servirgli."
    ),
}
READER_HEAD = {"en": "Notes from a colleague:\n", "it": "Appunti di un collega:\n"}
NO_NOTE = {"en": "(no notes)", "it": "(nessun appunto)"}


class Handoff(Pattern):
    name = "handoff"
    roles = ("writer", "reader")
    outcome = "reader"
    weights = {"writer": {"lift": 1.0, "readers": 0.0}, "reader": {"correct": 1.0}}
    hidden_options = ("prompt_field",)  # the question's column is question_field here
    fields = {
        "context": "the material only the writer sees (scenario.args.material_field)",
        "messages | prompt | question | problem": "the question only the reader sees (scenario.args.question_field)",
        "answer": "the reference answer, for the oracle partners (scenario.args.gold_field)",
        "language": "optional: en or it (scenario.args.language_field)",
    }

    def __init__(
        self,
        weights=None,
        material_field: str = "context",
        question_field: str = "",
        max_note_chars: int = 600,
        reader_samples: int = 2,
        control: bool = True,
        **options,
    ):
        if "prompt_field" in options:
            raise ValueError("handoff: the question's column is question_field (not prompt_field)")
        super().__init__(weights, prompt_field=question_field, **options)
        if reader_samples < 1 or max_note_chars < 1:
            raise ValueError("handoff: reader_samples and max_note_chars must be >= 1")
        self.material_field, self.max_note_chars = material_field, max_note_chars
        self.reader_samples, self.control = reader_samples, control
        if not control and "readers" not in ((weights or {}).get("writer") or {}):
            self.weights["writer"].update(readers=1.0, lift=0.0)  # no control, no lift: paid on its readers

    def _problem(self, row: dict, what: str, option: str) -> str:
        return super()._problem(row, what, "question_field" if option == "prompt_field" else option)

    def required(self, row: dict) -> None:
        super().required(row)
        self.text_field(row, self.material_field, "material_field")

    def advice(self, rows: list[dict]) -> list[str]:
        sizes = [len(row[self.material_field]) for row in rows if isinstance(row.get(self.material_field), str)]
        fit = sum(size <= self.max_note_chars for size in sizes)
        warnings = super().advice(rows)
        leaked = 0
        for row in rows:
            material = row.get(self.material_field)
            if isinstance(material, str) and material.strip()[:200] in self.task(row):
                leaked += 1
        if leaked:
            warnings.append(
                f"handoff: in {leaked} of {len(rows)} rows the question already contains the material, so the reader "
                "does not need the note. Point scenario.args.question_field at the column with the bare question."
            )
        if sizes and fit > len(sizes) / 2:
            warnings.append(
                f"handoff: {fit} of {len(sizes)} materials fit in a note of {self.max_note_chars} characters, so the "
                "best note is a copy. Lower scenario.args.max_note_chars."
            )
        return warnings

    # ------------------------------------------------------------------ prompts

    def writer_prompt(self, row: dict) -> str:
        material = self.text_field(row, self.material_field, "material_field").strip()
        return WRITER[self.language(row)].format(material=material, limit=self.max_note_chars)

    def _reader_tail(self, row: dict) -> str:
        return "\n\n" + self.with_answer_line(self.task(row), self.language(row))

    def reader_messages(self, row: dict, note: str) -> list[dict]:
        language = self.language(row)
        text = READER_HEAD[language] + (note.strip() or NO_NOTE[language]) + self._reader_tail(row)
        return [*self.conversation(row)[:-1], user(text)]

    # ------------------------------------------------------------------ partners

    def partners(self):
        def oracle_writer(messages, row):
            return self.gold(row) or ""

        def oracle_reader(messages, row):  # it answers only what the note supports
            note = between(messages[-1]["content"], READER_HEAD[self.language(row)], self._reader_tail(row)) or ""
            gold = self.gold(row)
            return self.say(gold) if gold is not None and gold.casefold() in note.casefold() else ""

        return {
            "writer": {
                "oracle": oracle_writer,
                "silent": lambda messages: "",
                "filler": lambda messages: "These are my notes on the text.",
            },
            "reader": {
                "oracle": oracle_reader,
                "silent": lambda messages: "",
                "refuse": lambda messages: "The notes do not say.",
            },
        }

    # ------------------------------------------------------------------ episode

    async def run(self, episode: Episode, row: dict) -> None:
        self.required(row)
        writer = await episode.run("writer", [user(self.writer_prompt(row))])
        channel = episode.channel("note", max_chars=self.max_note_chars)
        note = channel.send(writer.text if writer.ok else "", "writer", "reader")
        writer.reward("metric/note_chars", len(note))
        writer.reward("metric/note_cut", episode.log[-1]["cut"])

        async def read(branch: Episode):
            reader = await branch.run("reader", self.reader_messages(row, note))
            return (await self.pay(reader, row)).score

        async def unaided():
            control = await episode.run("reader", self.reader_messages(row, ""), slot="control")
            score = (await self.grade(self.answer_of(control), row, control)).score
            control.reward("metric/control", score)  # logged only: the control's session does not train
            return score

        if not self.control:
            writer.reward("readers", mean(await episode.branch(self.reader_samples, read, label="read")))
            return
        scores, baseline = await asyncio.gather(episode.branch(self.reader_samples, read, label="read"), unaided())
        readers = mean(scores)
        writer.reward("readers", readers)
        writer.reward("metric/control", baseline)
        writer.reward("lift", None if readers is None or baseline is None else readers - baseline)
