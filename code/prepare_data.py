import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TARGET = Path(__file__).resolve().parents[1] / "data"
SOURCES = {
    "hywiki": ("Data/wikipeople/wikipeople_kg.json", "Data/wikipeople/qa_v5.json", "Data/wikipeople"),
    "hywd": ("Data/wd50k/wd.txt", "Data/wd50k/qa_v3_all11_full.json", "Data/wd50k"),
    "kqapro": ("Data/kqapro_strict/kqapro_kg.csv", "Data/kqapro_strict/qa_val_strict.json", "Data/kqapro_strict"),
    "lcquad2": ("Data/lcquad2_20181217_local/lcquad2_20181217_source_neighborhood_kg.csv", "Data/lcquad2_20181217_local/qa_test_402.json", "Data/lcquad2_20181217_local"),
}


def as_list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--target", type=Path, default=TARGET)
    args = parser.parse_args()
    for name, (graph_path, qa_path, mapping_dir) in SOURCES.items():
        out = args.target / name
        out.mkdir(parents=True, exist_ok=True)
        graph = args.source_root / graph_path
        suffix = ".jsonl" if name == "hywiki" else ".csv"
        shutil.copyfile(graph, out / f"graph{suffix}")
        for mapping_name in ("entity_id2text.json", "relation_id2text.json"):
            shutil.copyfile(args.source_root / mapping_dir / mapping_name, out / mapping_name)
        qa = json.loads((args.source_root / qa_path).read_text(encoding="utf-8"))
        entities = json.loads((out / "entity_id2text.json").read_text(encoding="utf-8"))
        exported = []
        for row in qa:
            if name in {"hywiki", "hywd"}:
                sources = as_list(row.get("source"))
                path = row.get("path")
            elif name == "kqapro":
                sources = as_list(row.get("source_label"))
                path = None
            else:
                sources = [entities.get(str(value), str(value)) for value in as_list(row.get("kg_target_ids"))]
                path = None
            exported.append({
                "index": row["index"],
                "Question": row["question"],
                "Answer": as_list(row.get("answer")),
                "Source Entity": sources,
                "Reasoning Path": path,
            })
        if len({str(row["index"]) for row in exported}) != len(exported):
            raise ValueError(f"Duplicate question index in {name}")
        (out / "qa.json").write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(name, len(exported), graph.stat().st_size)


if __name__ == "__main__":
    main()
