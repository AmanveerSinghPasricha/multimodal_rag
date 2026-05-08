# test_memory.py
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from app.core.memory import MemoryManager


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.embed_query.return_value = [0.1] * 384
    emb.embed_documents.return_value = [[0.1] * 384]
    return emb


@pytest.fixture
def memory_manager(mock_embeddings, tmp_path):
    """Uses tmp_path so tests never write to the real 'memory/' directory."""
    with patch("app.core.memory.FAISS") as mock_faiss_cls:
        mock_store = MagicMock()
        mock_faiss_cls.from_texts.return_value = mock_store
        mock_store.similarity_search.return_value = []

        manager = MemoryManager(mock_embeddings, memory_path=str(tmp_path / "memory"))
        manager.memory = mock_store
        yield manager


def test_update_memory_stores_text(memory_manager):
    memory_manager.update_memory("User prefers concise answers")
    memory_manager.memory.add_texts.assert_called_once_with(["User prefers concise answers"])


def test_update_memory_skips_empty(memory_manager):
    memory_manager.update_memory("")
    memory_manager.memory.add_texts.assert_not_called()


def test_update_memory_skips_too_long(memory_manager):
    memory_manager.update_memory("x" * 400)
    memory_manager.memory.add_texts.assert_not_called()


def test_update_memory_blocks_sensitive(memory_manager):
    memory_manager.update_memory("patient has a diagnosis of diabetes")
    memory_manager.memory.add_texts.assert_not_called()


def test_get_long_memory_returns_string(memory_manager):
    memory_manager.memory.similarity_search.return_value = [
        Document(page_content="User likes bullet points", metadata={})
    ]
    result = memory_manager.get_long_memory("format preferences")
    assert isinstance(result, str)
    assert "User likes bullet points" in result


def test_get_long_memory_skips_init_text(memory_manager):
    memory_manager.memory.similarity_search.return_value = [
        Document(page_content="User initialized", metadata={})
    ]
    result = memory_manager.get_long_memory("anything")
    assert result == ""