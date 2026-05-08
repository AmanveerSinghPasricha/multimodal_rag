# test_rag.py
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.rag_engine import EnterpriseMultimodalRAG
from app.models.response import RAGResponse
from langchain_core.documents import Document


DUMMY_DOCS = [
    Document(
        page_content="The Transformer model uses self-attention to process sequences in parallel.",
        metadata={"source": "paper.pdf"},
    ),
    Document(
        page_content="Attention mechanisms allow the model to focus on relevant parts of the input.",
        metadata={"source": "paper.pdf"},
    ),
]

VALID_LLM_RESPONSE = json.dumps({
    "answer": "The Transformer uses self-attention to process sequences and allows focusing on relevant input parts.",
    "citations": ["paper.pdf"],
    "confidence": 0.85,
    "memory_update": None,
})


@pytest.fixture
def rag():
    with patch("app.core.rag_engine.ChatGroq"), \
         patch("app.core.rag_engine.HuggingFaceEndpointEmbeddings"), \
         patch("app.core.rag_engine.MemoryManager"):
        engine = EnterpriseMultimodalRAG()
        engine.embeddings.embed_query = MagicMock(return_value=[0.1] * 384)
        engine.memory_manager.get_long_memory = MagicMock(return_value="")
        engine.memory_manager.update_memory = MagicMock()
        return engine


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(return_value=DUMMY_DOCS)
    return retriever


@pytest.mark.asyncio
async def test_rag_basic(rag, mock_retriever):
    mock_response = MagicMock()
    mock_response.content = f"```json\n{VALID_LLM_RESPONSE}\n```"
    rag.llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await rag.run(
        query="How does the Transformer model work?",
        retriever=mock_retriever,
        history=[],
        image=None,
    )

    assert isinstance(result, RAGResponse)
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
    assert isinstance(result.citations, list)
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_rag_no_docs_returns_fallback(rag):
    retriever = MagicMock()
    retriever.ainvoke = AsyncMock(return_value=[])

    result = await rag.run(
        query="anything",
        retriever=retriever,
        history=[],
    )

    assert "No relevant data" in result.answer
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_rag_retries_on_bad_json(rag, mock_retriever):
    bad_response = MagicMock()
    bad_response.content = "This is not JSON at all"

    good_response = MagicMock()
    good_response.content = f"```json\n{VALID_LLM_RESPONSE}\n```"

    rag.llm.ainvoke = AsyncMock(side_effect=[bad_response, bad_response, good_response])

    result = await rag.run(
        query="How does attention work?",
        retriever=mock_retriever,
        history=[],
    )

    assert isinstance(result.answer, str)


def test_detect_intent_summary(rag):
    assert rag.detect_intent("give me a summary") == "summary"


def test_detect_intent_explanation(rag):
    assert rag.detect_intent("explain how this works") == "explanation"


def test_detect_intent_qa(rag):
    assert rag.detect_intent("what year was this published?") == "qa"


def test_is_valid_answer_rejects_short(rag):
    assert rag.is_valid_answer("Too short.") is False


def test_is_valid_answer_accepts_normal(rag):
    answer = "The model achieves state of the art results on multiple benchmarks using self-attention."
    assert rag.is_valid_answer(answer) is True