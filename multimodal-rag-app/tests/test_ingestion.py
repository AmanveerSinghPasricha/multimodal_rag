# test_ingestion.py
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.core.ingestion import ingest_file


DUMMY_DOCS = [
    Document(page_content="This is the first chunk of the document.", metadata={"source": "test.pdf"}),
    Document(page_content="This is the second chunk with more content.", metadata={"source": "test.pdf"}),
]


def test_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported file format"):
        ingest_file("some_file.xyz")


def test_ingestion_pdf():
    # FIX: PyPDFLoader is imported inside the function, so patch it at its
    # source location (langchain_community), not on the ingestion module.
    mock_loader_instance = MagicMock()
    mock_loader_instance.load.return_value = DUMMY_DOCS

    with patch("langchain_community.document_loaders.PyPDFLoader", return_value=mock_loader_instance), \
         patch("app.core.ingestion.TEXT_SPLITTER.split_documents", return_value=DUMMY_DOCS):
        docs = ingest_file("data/sample.pdf")

    assert docs is not None
    assert len(docs) > 0
    assert hasattr(docs[0], "page_content")


@patch("app.core.ingestion.pd.read_csv")
def test_ingestion_csv(mock_read_csv):
    import pandas as pd
    mock_read_csv.return_value = pd.DataFrame({
        "name": ["Alice", "Bob"],
        "score": [95, 82],
    })

    docs = ingest_file("data/data.csv")

    assert docs is not None
    assert len(docs) == 2
    assert "name: Alice" in docs[0].page_content