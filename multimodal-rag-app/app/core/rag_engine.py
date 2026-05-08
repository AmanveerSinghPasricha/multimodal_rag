import logging
import os
import functools
import base64
from typing import List, Optional

from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_pinecone import PineconeVectorStore
from langsmith import traceable

from app.models.response import RAGResponse
from app.utils.security import sanitize
from app.core.analytics_engine import CSVAnalytics 
from app.core.memory import MemoryManager
from app.core.ingestion import ingest_file 
from app.core.retrieval import UnifiedHybridRetriever

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

class EnterpriseMultimodalRAG:
    def __init__(self, namespace: str = "default"):
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1).with_structured_output(RAGResponse)
        self.fast_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        self.vision_llm = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")

        self.analytics_engine = CSVAnalytics(self.fast_llm)
        self.memory = MemoryManager(get_embeddings(), namespace=namespace)
        
        self.all_docs = [] 
        self.retriever = None 
        self.namespace = namespace
        self.active_parquet_path = None

        # 1. TOOL REGISTRY: Added "chat" for safe conversation routing
        self.tool_registry = {
            "analytics": "Best for mathematical operations, aggregations (sum/avg), statistical trends, or questions requiring a full scan of tabular data (CSV/Parquet).",
            "rag": "Best for semantic search and finding specific facts, text-based info, or explanations within document/PDF content.",
            "visual": "Best for analyzing image uploads, identifying objects in pictures, or describing visual diagrams.",
            "chat": "Best for general greetings, conversational questions, pleasantries, or asking about the AI's identity."
        }

        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        self.vectorstore = PineconeVectorStore(index=self.index, embedding=get_embeddings(), namespace=self.namespace)

    @traceable(name="Generalized Router")
    async def _route_query(self, query: str) -> str:
        """Determines intent based on tool capability descriptions."""
        # Dynamically build the prompt from the registry
        tool_desc = "\n".join([f"- {name}: {desc}" for name, desc in self.tool_registry.items()])
        
        routing_instructions = f"""
        Given the user query, select the most appropriate tool based on these descriptions:
        {tool_desc}
        
        User Query: "{query}"
        
        Return ONLY the word: 'analytics', 'visual', 'rag', or 'chat'.
        """
        
        res = await self.fast_llm.ainvoke([
            SystemMessage(content="You are a routing orchestrator. Output only the tool name."),
            HumanMessage(content=routing_instructions)
        ])
        return res.content.lower().strip()

    @traceable(name="Hallucination Guard")
    async def _is_query_analytical(self, query: str) -> bool:
        """Uses LLM reasoning to determine if a failed search should have been an analytical task."""
        check_prompt = (
            "The document search returned NO results. "
            "Does this query actually require a mathematical calculation or a scan of a data table? "
            f"Query: {query}"
        )
        res = await self.fast_llm.ainvoke([
            SystemMessage(content="Return ONLY 'yes' or 'no'."),
            HumanMessage(content=check_prompt)
        ])
        return res.content.lower().strip() == "yes"

    async def ingest(self, file_path: str):
        file_ns = os.path.basename(file_path).replace(".", "_")
        docs = await ingest_file(file_path, vision_llm=self.vision_llm)
        self.all_docs.extend(docs)
        await self.vectorstore.aadd_documents(docs, namespace=file_ns)

        if not self.retriever:
            self.retriever = UnifiedHybridRetriever(
                self.vectorstore.as_retriever(search_kwargs={"namespace": file_ns}), 
                self.all_docs
            )
        else:
            self.retriever.refresh_bm25(self.all_docs)

        for d in docs:
            if d.metadata.get("parquet_path"):
                self.active_parquet_path = d.metadata["parquet_path"]
        return docs

    @traceable(name="Main RAG Workflow")
    async def run(self, query: str, history: List[dict], image: Optional[str] = None) -> RAGResponse:
        clean_query = sanitize(query).lower()
        intent = await self._route_query(clean_query)
        past_context = await self.memory.get_relevant_context(clean_query)

        # --- CHITCHAT BYPASS WITH BOUNDARY GUARD ---
        if intent == "chat":
            chat_prompt = """
            You are Intelligence Studio, an advanced Agentic Multimodal Orchestration platform. 
            You were designed to bridge the trust gap in modern LLMs by providing zero-hallucination, 
            mathematically verifiable answers across diverse data types (PDFs, structured data, and images).
            
            STRICT RULES:
            1. You may answer greetings, pleasantries, and questions about your identity or purpose using the definition above.
            2. Keep your identity explanations professional, concise, and helpful.
            3. You MUST NOT answer ANY questions requiring facts, calculations, external data analysis, or document summaries.
            4. If the user asks a data/factual question, reply EXACTLY with: "I can help with that. Please ask your question again, and I will search your library and run the calculations."
            """
            combined_text = f"PAST:\n{past_context}\n\nQuery: {clean_query}"
            user_msg = HumanMessage(content=combined_text)
            
            response = await self.fast_llm.ainvoke([SystemMessage(content=chat_prompt), user_msg])
            
            await self.memory.update_memory(f"User: {clean_query} | AI: {response.content}")
            return RAGResponse(answer=response.content, confidence=1.0, citations=["System Greeting"])
        # -------------------------------------------

        if intent == "analytics" and self.active_parquet_path:
            raw_data = await self.analytics_engine.run_query(self.active_parquet_path, clean_query)
            answer, citations = str(raw_data), ["Analytics Engine"]
        else:
            docs = await self.retriever.ainvoke(clean_query) if self.retriever else []
            
            # 2. DYNAMIC FALLBACK: Catch routing errors without hardcoded word lists
            if not docs and self.active_parquet_path:
                if await self._is_query_analytical(clean_query):
                    logger.warning("Empty RAG result for analytical query. Forcing Analytics Engine fallback.")
                    raw_data = await self.analytics_engine.run_query(self.active_parquet_path, clean_query)
                    return RAGResponse(answer=str(raw_data), confidence=1.0, citations=["Analytics Engine (Fallback)"])

            doc_context = "\n\n".join([f"[{d.metadata.get('universal_citation')}]: {d.page_content}" for d in docs])
            system_prompt = "Use context to answer. Base definitions ONLY on DOCUMENT CONTEXT."
            combined_text = f"PAST:\n{past_context}\n\nDOCS:\n{doc_context}\n\nQuery: {clean_query}"
            
            if image:
                with open(image, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                user_msg = HumanMessage(content=[
                    {"type": "text", "text": combined_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
                ])
                response = await self.vision_llm.ainvoke([SystemMessage(content=system_prompt), user_msg])
                answer = response.content
            else:
                user_msg = HumanMessage(content=combined_text)
                response = await self.llm.ainvoke([SystemMessage(content=system_prompt), user_msg])
                answer = response.answer

            citations = list(set([d.metadata.get("universal_citation", "Unknown") for d in docs]))

        await self.memory.update_memory(f"User: {clean_query} | AI: {answer}")
        return RAGResponse(answer=answer, confidence=1.0, citations=citations)