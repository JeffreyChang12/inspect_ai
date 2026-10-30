import os
from pathlib import Path

import pytest
from test_helpers.tasks import popularity

from inspect_ai import Task, eval, eval_retry, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog
from inspect_ai.solver import Generate, TaskState, solver


def test_sample_shuffle():
    check_sample_shuffle()


def test_sample_shuffle_limit():
    check_sample_shuffle(limit=20)


def test_sample_shuffle_no_id():
    with pytest.raises(ValueError):
        check_sample_shuffle(sample_id=[1, 2, 3, 4, 5])


def test_sample_no_shuffle():
    ids = range(100, 200)
    dataset = [Sample(id=id, input=f"Do you like the id '{id}'?") for id in ids]
    log1 = eval(Task(dataset=dataset), model="mockllm/model", limit=20)[0]
    log2 = eval(Task(dataset=dataset), model="mockllm/model", limit=20)[0]
    check_samples_equal(log1, log2)


def test_sample_shuffle_seed_zero():
    """0 is a seed like any other int, giving the same order as `shuffle(0)`."""
    samples = [Sample(id=id, input=f"Question {id}") for id in range(1, 21)]
    seeded = MemoryDataset(list(samples))
    seeded.shuffle(0)

    shuffle_task = Task(dataset=samples)
    log = eval(shuffle_task, model="mockllm/model", sample_shuffle=0, limit=4)[0]

    assert log.eval.dataset.shuffled
    assert log.eval.dataset.sample_ids == [sample.id for sample in seeded][:4]


FAIL_ENV = "INSPECT_TEST_SAMPLE_SHUFFLE_FAIL"


@solver
def fail_while_env_set():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if os.environ.get(FAIL_ENV) == "1":
            raise RuntimeError("Transient failure")
        return state

    return solve


@task
def sample_shuffle_retry_task() -> Task:
    return Task(
        dataset=[Sample(id=id, input=f"Question {id}") for id in range(1, 21)],
        solver=fail_while_env_set(),
    )


@pytest.mark.parametrize("sample_shuffle", [None, False, 0, 42])
def test_sample_shuffle_retry_reruns_same_samples(
    sample_shuffle: bool | int | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """eval_retry() selects the same samples as the run it retries when there is a limit."""
    monkeypatch.setenv(FAIL_ENV, "1")
    log = eval(
        sample_shuffle_retry_task(),
        model="mockllm/model",
        sample_shuffle=sample_shuffle,
        limit=4,
        log_dir=str(tmp_path),
    )[0]
    assert log.status == "error"

    monkeypatch.delenv(FAIL_ENV)
    retried = eval_retry(log, log_dir=str(tmp_path))[0]

    assert retried.status == "success"
    assert retried.eval.dataset.sample_ids == log.eval.dataset.sample_ids
    assert retried.samples
    assert {sample.id for sample in retried.samples} == set(
        log.eval.dataset.sample_ids or []
    )


def check_sample_shuffle(**kwargs):
    log1 = eval(popularity(), model="mockllm/model", sample_shuffle=42, **kwargs)[0]
    log2 = eval(popularity(), model="mockllm/model", sample_shuffle=42, **kwargs)[0]
    check_samples_equal(log1, log2)


def check_samples_equal(log1: EvalLog, log2: EvalLog) -> None:
    assert log1.samples and log2.samples
    assert [sample.input for sample in log1.samples] == [
        sample.input for sample in log2.samples
    ]
