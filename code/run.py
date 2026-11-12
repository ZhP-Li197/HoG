import argparse
import json
from pathlib import Path

from embeddings import Encoders
from graph import Graph
from hog import HoG
from llm import LLM


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("hywiki", "hywd", "kqapro", "lcquad2"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--entity-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--statement-encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url")
    args = parser.parse_args()
    directory = ROOT / "data" / args.dataset
    questions = json.loads((directory / "qa.json").read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[:args.limit]
    output = args.output or ROOT / "outputs" / f"{args.dataset}_{args.model.replace('/', '_')}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            latest = {str(item["index"]): item for line in handle if line.strip() for item in [json.loads(line)]}
            completed = {index for index, item in latest.items() if "error" not in item}
    graph = Graph(directory)
    llm = LLM(args.model, base_url=args.base_url)
    encoders = Encoders(args.entity_encoder, args.statement_encoder)
    agent = HoG(graph, llm, encoders, retrieval_k=10, pruning_k=3)
    with output.open("a", encoding="utf-8") as handle:
        for position, row in enumerate(questions):
            if str(row["index"]) in completed:
                continue
            try:
                result = agent.solve(row["Question"])
                record = {"index": row["index"], "Question": row["Question"], **result}
            except Exception as exc:
                record = {"index": row["index"], "Question": row["Question"], "error": f"{type(exc).__name__}: {exc}"}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(position, row["index"], record.get("answer", record.get("error")), flush=True)


if __name__ == "__main__":
    main()
