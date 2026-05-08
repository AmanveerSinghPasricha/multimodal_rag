import asyncio
import logging

from app.core.rag_engine import EnterpriseMultimodalRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# QUERY REWRITE
# =========================
def rewrite_query(query: str) -> str:
    query = query.strip()

    vague_patterns = [
        "explain the document",
        "summarize",
        "what is this",
        "describe this",
    ]

    if any(p in query.lower() for p in vague_patterns):
        return (
            "Explain in detail the key technical concepts, components, "
            "and working mechanisms described in the document."
        )

    return query


# =========================
# RETRIEVAL DEBUG
# =========================
def debug_retrieval(query: str, docs: list) -> None:
    print("\n================ RETRIEVED DOCS ================\n")
    contents = []

    for i, d in enumerate(docs):
        content = d.page_content.strip()
        contents.append(content)
        print(f"\n--- DOC {i+1} ---")
        print("SOURCE:", d.metadata.get("source"))
        print("CONTENT:", content[:500])

    unique_count = len(set(contents))
    print(f"\nUnique docs: {unique_count} / {len(contents)}")

    matched = sum(
        1 for d in docs
        if any(w.lower() in d.page_content.lower() for w in query.split())
    )
    score = matched / max(1, len(docs))
    print(f"Retrieval keyword coverage: {score:.2f}")


# =========================
# MAIN
# =========================
async def main():
    rag = EnterpriseMultimodalRAG()

    file_path = input("Enter path to your document (e.g. data/paper.pdf): ").strip()
    retriever = rag.ingest(file_path)

    history = []

    print("\nType your question below. Type 'quit' to exit.\n")

    while True:
        user_query = input("You: ").strip()
        if user_query.lower() in {"quit", "exit", "q"}:
            break
        if not user_query:
            continue

        final_query = rewrite_query(user_query)
        if final_query != user_query:
            print(f"\n[Query rewritten → {final_query}]\n")

        # Debug: inspect retrieved docs before LLM call
        docs = await retriever.ainvoke(final_query)
        debug_retrieval(final_query, docs)

        result = await rag.run(
            query=final_query,
            retriever=retriever,
            history=history,
            image=None,
        )

        print("\n================ ANSWER ================\n")
        print(result.answer)
        print("\nCitations:", list(set(result.citations)))
        print(f"Confidence: {result.confidence:.2f}")
        print()

        # Maintain rolling history for multi-turn conversation
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    asyncio.run(main())