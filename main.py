from pathlib import Path
import argparse
import json
from rag.ingest import IngestPipeline
from rag.retriever import Retriever
from rag.prompt_builder import PromptBuilder
from rag.llm_responder import LLMResponder


def save_results_to_json(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="CodeValueRAG CLI")
    parser.add_argument("--repo", type=str, default="./data/ODrive", help="Path to code repository")
    parser.add_argument("--query", type=str, default="How does the PID controller work?", help="Query for retrieval")
    parser.add_argument("--index", type=str, default="index/faiss.index", help="Path to index file")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON file")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing if index already exists")
    parser.add_argument("--show-code", choices=["none", "full", "diff"], default="full", help="Show code content")
    parser.add_argument("--llm", action="store_true", help="Use LLM to generate answer")
    parser.add_argument("--model-name", type=str, default="tiiuae/falcon-7b-instruct", help="LLM model name")
    args = parser.parse_args()

    print(f"{args=}")

    if not args.skip_index or not Path(args.index).exists():
        print("[Indexing] Running IngestPipeline...")
        pipeline = IngestPipeline(repo_root=Path(args.repo), index_path=Path(args.index))
        pipeline.run()

    print(f"[Retrieving] Query: {args.query}")
    retriever = Retriever(index_path=args.index)
    results = retriever.retrieve(args.query)

    # 結果を表示
    for res in results:
        print(f"{res['score']:.4f} - {res['file_path']}#{res['symbol']}")

    save_results_to_json(results, args.output)
    print(f"[Saved] Results saved to: {args.output}")

    # コードの表示
    if args.show_code != "none":
        from rag.display import show_full_code, show_diff
        if args.show_code == "full":
            show_full_code(results)
        elif args.show_code == "diff":
            show_diff(args.query, results)

    # LLMによる回答生成
    if args.llm:
        print(f"[LLM] Generating answer with: {args.model_name}")
        builder = PromptBuilder()
        prompt = builder.build(args.query, results)
        responder = LLMResponder(model_name=args.model_name)
        answer = responder.generate(prompt)
        print("\n[LLM Answer]")
        print(answer)


if __name__ == "__main__":
    main()
