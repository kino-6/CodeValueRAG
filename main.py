from pathlib import Path
import argparse
import json
import logging
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
    parser.add_argument("--model-name", type=str, default="codellama/CodeLlama-7b-Instruct-hf", help="LLM model name")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                       default="INFO", help="Set the logging level")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.debug(f"Arguments: {args}")

    if not args.skip_index or not Path(args.index).exists():
        logger.info("[Indexing] Running IngestPipeline...")
        pipeline = IngestPipeline(repo_root=Path(args.repo), index_path=Path(args.index))
        pipeline.run()

    logger.info(f"[Retrieving] Query: {args.query}")
    retriever = Retriever(index_path=args.index)
    results = retriever.retrieve(args.query)

    # 検索結果の表示（ログレベルに関係なく表示）
    print("\n=== Search Results ===")
    for res in results:
        print(f"{res['score']:.4f} - {res['file_path']}#{res['symbol']}")
    print("=====================\n")

    save_results_to_json(results, args.output)
    logger.info(f"[Saved] Results saved to: {args.output}")

    # コードの表示（ログレベルに関係なく表示）
    if args.show_code != "none":
        from rag.display import show_full_code, show_diff
        if args.show_code == "full":
            show_full_code(results)
        elif args.show_code == "diff":
            show_diff(args.query, results)

    # LLMによる回答生成（ログレベルに関係なく表示）
    if args.llm:
        logger.info(f"[LLM] Generating answer with: {args.model_name}")
        builder = PromptBuilder()
        prompt = builder.build(args.query, results)
        responder = LLMResponder(model_name=args.model_name)
        answer = responder.generate(prompt)
        print("\n=== LLM Answer ===")
        print(answer)
        print("=================\n")


if __name__ == "__main__":
    main()
