import asyncio
from types import SimpleNamespace

import agent


class _FakeEvent:
    def __init__(self, *, author: str, text: str, partial: bool = False, final: bool = False):
        self.author = author
        self.partial = partial
        self._final = final
        self.content = SimpleNamespace(parts=[SimpleNamespace(text=text)])

    def is_final_response(self) -> bool:
        return self._final


class _FakeRunner:
    def __init__(self, events):
        self._events = list(events)

    async def run_async(self, *, session_id, user_id, new_message):  # noqa: ARG002
        for event in self._events:
            yield event


def test_run_native_workflow_turn_ignores_non_final_benchmark_scratch():
    runner = _FakeRunner([
        _FakeEvent(author="benchmark_executor", text="I need to call the tool first."),
        _FakeEvent(author="benchmark_executor", text="APOE: 0.6105 APOC1: 0.5323 APOC2: 0.5324", final=True),
    ])

    answer = asyncio.run(
        agent._run_native_workflow_turn(
            runner,
            session_id="s1",
            user_id="u1",
            prompt="What is the GC content?",
        )
    )

    assert answer == "APOE: 0.6105 APOC1: 0.5323 APOC2: 0.5324"


def test_run_native_workflow_turn_still_concatenates_true_partial_chunks():
    runner = _FakeRunner([
        _FakeEvent(author="report_synthesizer", text="Part-one-", partial=True),
        _FakeEvent(author="report_synthesizer", text="part-two", final=True),
    ])

    answer = asyncio.run(
        agent._run_native_workflow_turn(
            runner,
            session_id="s1",
            user_id="u1",
            prompt="Summarize this.",
        )
    )

    assert answer == "Part-one-part-two"
