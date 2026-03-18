import argparse
import os

from ruamel.yaml import YAML


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a coding config that uses AMemMemory."
    )
    parser.add_argument("--source-config", required=True, help="Source YAML config.")
    parser.add_argument("--target-config", required=True, help="Target YAML config.")
    parser.add_argument("--llm-model", default=None, help="Override top-level LLM.")
    parser.add_argument(
        "--evaluate-llm",
        default="bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Override evaluation model.",
    )
    parser.add_argument(
        "--workspace-dir",
        required=True,
        help="Workspace dir relative to marble/.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output jsonl path relative to marble/.",
    )
    parser.add_argument(
        "--embedding-model",
        default="bedrock/amazon.titan-embed-text-v2:0",
        help="Embedding model for AMemMemory.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=3,
        help="Number of candidate notes to retrieve per query.",
    )
    parser.add_argument(
        "--max-memory-context",
        type=int,
        default=5,
        help="Maximum notes included in prompt memory context.",
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=0.72,
        help="Cosine similarity threshold for note linking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.source_config):
        raise FileNotFoundError(f"Source config not found: {args.source_config}")

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(args.source_config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f)

    if args.llm_model:
        cfg["llm"] = args.llm_model

    cfg["environment"]["workspace_dir"] = args.workspace_dir
    cfg["environment"]["max_iterations"] = 3
    if "metrics" not in cfg or cfg["metrics"] is None:
        cfg["metrics"] = {}
    cfg["metrics"]["evaluate_llm"] = args.evaluate_llm
    cfg["memory"] = {
        "type": "AMemMemory",
        "embedding_model": args.embedding_model,
        "retrieval_top_k": args.retrieval_top_k,
        "max_memory_context": args.max_memory_context,
        "link_threshold": args.link_threshold,
    }
    cfg["output"]["file_path"] = args.output_path

    os.makedirs(os.path.dirname(args.target_config), exist_ok=True)
    with open(args.target_config, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    print(f"Prepared A-Mem config: {args.target_config}")


if __name__ == "__main__":
    main()
