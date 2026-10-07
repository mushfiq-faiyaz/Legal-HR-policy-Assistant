"""
Quick sanity check for the vector store built by ingest.py.
Tests retrieval across legal and HR policy documents without calling Groq LLM.

Usage:
    python test_query.py --query "What is the notice period for termination?"
"""

import argparse
import chromadb
from chromadb.utils import embedding_functions


def main():
    parser = argparse.ArgumentParser(description="Test ChromaDB retrieval for Legal/HR policies.")
    parser.add_argument("--db_dir", default="./chroma_db", help="Path to Chroma DB folder")
    parser.add_argument("--collection_name", default="nexora", help="Chroma collection name")
    parser.add_argument("--query", required=True, help="Question or clause to search for")
    parser.add_argument("--top_k", type=int, default=4, help="Number of chunks to retrieve")
    args = parser.parse_args()

    print(f"Searching collection '{args.collection_name}' for query: '{args.query}'\\n")

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=args.db_dir)

    try:
        collection = client.get_collection(args.collection_name, embedding_function=embed_fn)
    except Exception as e:
        print(f"Error: Collection '{args.collection_name}' not found in {args.db_dir}. Run ingest.py first.")
        return

    results = collection.query(query_texts=[args.query], n_results=args.top_k)

    print(f"=== Top {args.top_k} Retrieved Chunks ===")
    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ):
        source = meta.get("source", "Unknown")
        section = meta.get("section", "Unknown Section")
        page = meta.get("start_page", "?")
        print(f"\\n[Chunk {i+1}] Source: {source} | Section: {section} | Page: {page} | Distance: {dist:.4f}")
        print(f"Excerpt: {doc[:200]}...")


if __name__ == "__main__":
    main()
