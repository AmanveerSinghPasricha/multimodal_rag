import hashlib
import asyncio
import logging
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.documents import Document
from langsmith import traceable # Added for unified tracing

logger = logging.getLogger(__name__)

@traceable(name="Hybrid RRF Merge") # NEW: Trace the merging logic
def rrf_merge(bm25_docs: List[Document], vector_docs: List[Document], k: int = 60):
    """Reciprocal Rank Fusion for merging search results."""
    scores = {}
    doc_map = {}
    
    for stream in [bm25_docs, vector_docs]:
        for rank, doc in enumerate(stream):
            # Prioritize chunk_id from metadata for consistent merging
            key = doc.metadata.get("chunk_id") or hashlib.md5(doc.page_content.encode()).hexdigest()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc
            
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]

class UnifiedHybridRetriever:
    def __init__(self, vector_retriever, docs: List[Document]):
        self.vector = vector_retriever
        self.reranker = FlashrankRerank(top_n=5)
        self.relevance_threshold = 0.1 
        self._init_bm25(docs)

    def _init_bm25(self, docs: List[Document]):
        """Initializes or refreshes the static BM25 index."""
        if not docs:
            logger.warning("BM25 initialized with no documents. Using fallback.")
            docs = [Document(page_content="Placeholder", metadata={"chunk_id": "init"})]
        
        self.bm25 = BM25Retriever.from_documents(docs)
        self.bm25.k = 15 

    def refresh_bm25(self, updated_docs: List[Document]):
        """Re-initializes BM25 when new files are ingested."""
        logger.info(f"Refreshing BM25 with {len(updated_docs)} documents.")
        self._init_bm25(updated_docs)

    @traceable(name="Unified Hybrid Retrieval") # NEW: Links this call to the parent trace
    async def ainvoke(self, query: str) -> List[Document]:
        """Parallel retrieval with reranking and thresholding."""
        try:
            # Parallel execution of Vector and BM25 searches
            # These will now appear as sub-tasks in the LangSmith trace
            v_task = self.vector.ainvoke(query)
            b_task = asyncio.to_thread(self.bm25.invoke, query)
            
            v_docs, b_docs = await asyncio.gather(v_task, b_task)
            
            # Step 1: Hybrid Merge using RRF
            merged_docs = rrf_merge(b_docs, v_docs)
            if not merged_docs:
                return []

            # Step 2: Contextual Reranking (Using Flashrank)
            # traceable will capture the latency of this specific compression step
            reranked_docs = await asyncio.to_thread(
                self.reranker.compress_documents, merged_docs[:15], query
            )
            
            # Step 3: Thresholding based on relevance score
            final_docs = [
                doc for doc in reranked_docs 
                if doc.metadata.get("relevance_score", 1.0) >= self.relevance_threshold
            ]
            
            return final_docs
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []