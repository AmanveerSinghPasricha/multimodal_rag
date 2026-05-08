import os
import asyncio
import logging
import time
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, embeddings, namespace: str = "default"):
        self.memory_path = f"memory/{namespace}"
        self.embeddings = embeddings
        self.memory = None
        self._lock = asyncio.Lock()
        self._init_memory()

    def _init_memory(self):
        """Initializes local persistent memory for the user session."""
        os.makedirs(self.memory_path, exist_ok=True)
        index_file = os.path.join(self.memory_path, "index.faiss")

        if os.path.exists(index_file):
            try:
                self.memory = FAISS.load_local(
                    self.memory_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                return
            except Exception as e:
                logger.error(f"Memory load failed: {e}")

        # Seed node to establish vector dimensions
        init_doc = Document(
            page_content="SYSTEM_START", 
            metadata={"timestamp": time.time(), "tier": "system"}
        )
        self.memory = FAISS.from_documents([init_doc], self.embeddings)
        self.memory.save_local(self.memory_path)

    async def update_memory(self, text: str):
        """Stores meaningful interactions while avoiding near-exact duplicates."""
        if not text or len(text) < 5: return

        async with self._lock:
            try:
                # Semantic deduplication check
                existing = await asyncio.to_thread(
                    self.memory.similarity_search_with_score, text, k=1
                )
                if existing and existing[0][1] < 0.1: return

                new_doc = Document(
                    page_content=text,
                    metadata={"timestamp": time.time()}
                )
                await asyncio.to_thread(self.memory.add_documents, [new_doc])
                await asyncio.to_thread(self.memory.save_local, self.memory_path)
            except Exception as e:
                logger.error(f"Memory update failed: {e}")

    async def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Retrieves past interactions biased toward recency (Temporal Weighting)."""
        try:
            # Fetch a larger pool to allow for time-based re-sorting
            results = await asyncio.to_thread(
                self.memory.similarity_search_with_score, query, k=k*2
            )
            
            # Filter relevance and exclude system nodes
            valid = [
                (doc, score) for doc, score in results 
                if score < 0.75 and doc.page_content != "SYSTEM_START"
            ]

            # Sort by timestamp: Newest memories first
            valid.sort(key=lambda x: x[0].metadata.get("timestamp", 0), reverse=True)
            return "\n".join([doc.page_content for doc, _ in valid[:k]])
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return ""

    async def clear_memory(self):
        """Wipes the local memory index."""
        async with self._lock:
            try:
                import shutil
                if os.path.exists(self.memory_path):
                    await asyncio.to_thread(shutil.rmtree, self.memory_path)
                self._init_memory()
            except Exception as e:
                logger.error(f"Clear memory failed: {e}")