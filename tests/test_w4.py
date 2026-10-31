"""เป้าหมาย W4: pytest -m w4"""
import pytest

from rag.evaluate import evaluate, load_questions
from rag.retrieve import Retriever

pytestmark = pytest.mark.w4


@pytest.fixture(scope="module")
def dev_metrics():
    return evaluate(Retriever(method="hybrid"), load_questions("dev"), k=4, include_superseded=False)


def test_version_questions_get_latest(dev_metrics):
    assert dev_metrics["version_top1_active"] == 1.0


def test_hit_at_1_improves(dev_metrics):
    assert dev_metrics["hit@1"] >= 0.9
