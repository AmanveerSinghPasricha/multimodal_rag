import os
import hashlib
import logging
import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

logger = logging.getLogger(__name__)

def robust_normalize(text: str) -> str:
    """Cleans text while preserving technical units and symbols."""
    import re
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Protect technical units (e.g., 400kV, 10GB)
    text = re.sub(r'(\d+)\s?([kKmMgG][vVwWbB])', r'\1\2', text)
    return text.strip()

async def ingest_file(file_path: str, vision_llm=None) -> List[Document]:
    """Processes PDF/CSV files into enriched, cited chunks."""
    docs = []
    file_name = os.path.basename(file_path)

    if file_path.endswith('.pdf'):
        loader = PyMuPDFLoader(file_path)
        raw_docs = await asyncio.to_thread(loader.load)
        
        for i, doc in enumerate(raw_docs):
            page_num = doc.metadata.get("page", i + 1)
            content = robust_normalize(doc.page_content)
            
            if len(content) < 20: continue # Skip empty/noise pages

            # Assign Deep Metadata for Retrieval Traceability
            doc.metadata.update({
                "universal_citation": f"{file_name} (Pg {page_num})",
                "chunk_id": hashlib.md5(f"{file_name}_{page_num}_{content[:50]}".encode()).hexdigest(),
                "source": file_name
            })
            doc.page_content = content
            docs.append(doc)
            
    elif file_path.endswith(('.csv', '.parquet', '.xlsx')):
        # For data files, we track the path for the Analytics Engine
        doc = Document(
            page_content=f"Dataset reference for {file_name}",
            metadata={
                "universal_citation": "Data Analytics Engine",
                "parquet_path": file_path,
                "chunk_id": hashlib.md5(file_path.encode()).hexdigest()
            }
        )
        docs.append(doc)

    logger.info(f"Ingested {len(docs)} chunks from {file_name}")
    return docs