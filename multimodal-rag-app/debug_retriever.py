import os
import asyncio
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.retrieval import UnifiedHybridRetriever
from langchain_core.documents import Document

# 1. Setup Environment
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

async def debug_retrieval():
    print("--- 🛠 Starting RAG Debugger ---")
    
    # 2. Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    
    # 3. Connect to Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)
    
    # Use the same namespace as your App (usually 'default' or the filename)
    namespace = "default" 
    
    vectorstore = PineconeVectorStore(
        index=index, 
        embedding=embeddings, 
        namespace=namespace
    )

    print(f"✅ Connected to Index: {index_name} | Namespace: {namespace}")

    # 4. Check Raw Pinecone Count
    stats = index.describe_index_stats()
    ns_stats = stats.get('namespaces', {}).get(namespace, {})
    record_count = ns_stats.get('vector_count', 0)
    print(f"📊 Total Records in '{namespace}': {record_count}")

    if record_count == 0:
        print("❌ ERROR: No records found in this namespace. Ingestion failed or namespace mismatch.")
        return

    # 5. Test Vector Search Directly
    query = "self attention" # Change this to a term in your PDF
    print(f"\n🔍 Testing Raw Vector Search for: '{query}'...")
    v_results = await vectorstore.asimilarity_search(query, k=3)
    
    if not v_results:
        print("❌ ERROR: Vector search returned nothing. Check your embeddings or query.")
    else:
        print(f"✅ Found {len(v_results)} chunks via Vector Search.")
        for i, doc in enumerate(v_results):
            print(f"   [{i+1}] {doc.metadata.get('universal_citation', 'No Cite')}: {doc.page_content[:60]}...")

    # 6. Test Hybrid Retriever (Requires 'all_docs' for BM25)
    # Note: In production, we'd fetch all docs from Pinecone or local storage
    # For debugging, we'll create a dummy list to see if the object initializes
    print(f"\n🧠 Testing Hybrid Retriever Logic...")
    sample_docs = [Document(page_content="initialization", metadata={"chunk_id": "1"})]
    retriever = UnifiedHybridRetriever(vectorstore.as_retriever(), sample_docs)
    
    h_results = await retriever.ainvoke(query)
    print(f"✅ Hybrid/Reranked results returned: {len(h_results)}")

if __name__ == "__main__":
    asyncio.run(debug_retrieval())