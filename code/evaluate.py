import argparse
import json
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).replace("_", " ").casefold().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("hywiki", "hywd", "kqapro", "lcquad2"))
    parser.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()
    questions = json.loads((ROOT / "data" / args.dataset / "qa.json").read_text(encoding="utf-8"))
    predictions = {}
    with args.predictions.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                predictions[str(row["index"])] = row
    exact = 0
    f1_total = 0.0
    missing = 0
    errors = 0
    for row in questions:
        result = predictions.get(str(row["index"]))
        gold = {normalize(value) for value in row["Answer"] if normalize(value)}
        if result is None:
            missing += 1
            predicted = set()
        else:
            errors += int(bool(result.get("error")))
            predicted = {normalize(value) for value in result.get("answer", {}).get("answers", []) if normalize(value)}
        exact += int(bool(gold) and predicted == gold)
        f1_total += 2 * len(predicted & gold) / (len(predicted) + len(gold)) if predicted or gold else 0.0
    total = len(questions)
    print(json.dumps({"dataset": args.dataset, "questions": total, "missing": missing, "errors": errors, "exact_correct": exact, "EM": exact / total, "macro_F1": f1_total / total}, indent=2))


if __name__ == "__main__":
    main()
