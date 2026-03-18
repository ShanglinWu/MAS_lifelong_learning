import argparse
import os

from ruamel.yaml import YAML


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a database config that uses AMemMemory."
    )
    parser.add_argument("--source-config", required=True, help="Source YAML config.")
    parser.add_argument("--target-config", required=True, help="Target YAML config.")
    parser.add_argument("--llm-model", required=True, help="Override top-level LLM.")
    parser.add_argument(
        "--evaluate-llm",
        default="bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Override evaluation model.",
    )
    parser.add_argument(
        "--agent-count",
        type=int,
        default=3,
        help="Number of agents to keep in the graph.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Output JSON path relative to marble/.",
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
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum graph iterations.",
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

    cfg["coordinate_mode"] = "graph"
    cfg["communication"] = True
    cfg["llm"] = args.llm_model

    cfg.setdefault("environment", {})
    cfg["environment"]["type"] = "DB"
    cfg["environment"]["name"] = "DB Simulation Environment"
    cfg["environment"]["max_iterations"] = args.max_iterations

    agents = cfg.get("agents", [])
    cfg["agents"] = agents[: args.agent_count]
    selected_agent_ids = {agent["agent_id"] for agent in cfg["agents"]}

    relationships = cfg.get("relationships", [])
    cfg["relationships"] = [
        relation
        for relation in relationships
        if len(relation) >= 2
        and relation[0] in selected_agent_ids
        and relation[1] in selected_agent_ids
    ]

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

    cfg.setdefault("output", {})
    cfg["output"]["file_path"] = args.output_path

    os.makedirs(os.path.dirname(args.target_config), exist_ok=True)
    with open(args.target_config, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    print(f"Prepared A-Mem database config: {args.target_config}")


if __name__ == "__main__":
    main()
