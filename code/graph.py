import csv
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).replace("_", " ").casefold().split())


def forms(value):
    raw = norm(value)
    result = {raw}
    match = re.match(r"^([+-]?\d{3,6})[-/](\d{1,2})[-/](\d{1,2})", raw)
    if match:
        year, month, day = match.groups()
        result.update({year.lstrip("+"), f"{year.lstrip('+')}-{month.zfill(2)}", f"{year.lstrip('+')}-{month.zfill(2)}-{day.zfill(2)}"})
    match = re.match(r"^([+-]?\d{3,6})[-/]", raw)
    if match:
        result.add(match.group(1).lstrip("+"))
    return result


@dataclass(frozen=True)
class Statement:
    sid: int
    head: str
    relation: str
    tail: str
    qualifiers: tuple


class Graph:
    def __init__(self, directory):
        directory = Path(directory)
        self.entities = json.loads((directory / "entity_id2text.json").read_text(encoding="utf-8"))
        self.relations = json.loads((directory / "relation_id2text.json").read_text(encoding="utf-8"))
        self.entity_ids = defaultdict(list)
        self.relation_ids = defaultdict(list)
        for key, label in self.entities.items():
            self.entity_ids[norm(label)].append(key)
        for key, label in self.relations.items():
            self.relation_ids[norm(label)].append(key)
        self.statements = []
        self.head = defaultdict(set)
        self.tail = defaultdict(set)
        self.qualifier_value = defaultdict(set)
        self.qualifier_key = defaultdict(set)
        self.relation = defaultdict(set)
        graph_files = list(directory.glob("graph.*"))
        if len(graph_files) != 1:
            raise ValueError(f"Expected one graph file in {directory}")
        with graph_files[0].open(encoding="utf-8", newline="") as handle:
            if graph_files[0].suffix == ".jsonl":
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        for key in row:
                            if key.endswith("_h") and f"{key[:-2]}_t" in row:
                                relation = key[:-2]
                                qualifiers = tuple((k, str(v)) for k, values in row.items() if k not in {"N", key, f"{relation}_t"} and not k.endswith(("_h", "_t")) and isinstance(values, list) for v in values)
                                self._add(str(row[key]), relation, str(row[f"{relation}_t"]), qualifiers)
            else:
                for row in csv.reader(handle):
                    if len(row) < 3 or len(row[3:]) % 2:
                        raise ValueError("Malformed graph statement")
                    self._add(row[0], row[1], row[2], tuple((row[i], row[i + 1]) for i in range(3, len(row), 2)))

    def label(self, entity):
        return self.entities.get(entity, entity)

    def relation_label(self, relation):
        return self.relations.get(relation, relation)

    def _value_forms(self, value):
        return forms(value) | forms(self.label(value))

    def _add(self, head, relation, tail, qualifiers):
        sid = len(self.statements)
        statement = Statement(sid, head, relation, tail, qualifiers)
        self.statements.append(statement)
        self.head[head].add(sid)
        self.tail[tail].add(sid)
        self.relation[relation].add(sid)
        for key, value in qualifiers:
            self.qualifier_key[key].add(sid)
            self.qualifier_value[value].add(sid)

    def entity_candidates(self, text):
        if text in self.entities:
            return [text]
        return self.entity_ids.get(norm(text), [])

    def relation_candidates(self, text):
        if text in self.relations:
            return [text]
        return self.relation_ids.get(norm(text), [])

    def incident(self, entity):
        return self.head[entity] | self.tail[entity] | self.qualifier_value[entity]

    def serialized(self, statement):
        head = self.label(statement.head)
        relation = self.relation_label(statement.relation)
        tail = self.label(statement.tail)
        pairs = "; ".join(f"{self.relation_label(key)} = {self.label(value)}" for key, value in statement.qualifiers)
        return f"{head} -- {relation} --> {tail}" + (f" [{pairs}]" if pairs else "")

    def _condition(self, statement, condition):
        key = str(condition.get("key", ""))
        value = str(condition.get("value", ""))
        keys = self.relation_candidates(key)
        return any((not key or qkey in keys or norm(self.relation_label(qkey)) == norm(key)) and (not value or bool(self._value_forms(qvalue) & forms(value))) for qkey, qvalue in statement.qualifiers)

    def matches(self, relation, entity, direction, hypothesis):
        relations = self.relation_candidates(relation)
        if not relations:
            raise ValueError(f"Unknown relation: {relation}")
        if hypothesis:
            pools = []
            for condition in hypothesis:
                keys = self.relation_candidates(str(condition.get("key", "")))
                if not keys:
                    return
                pools.append(set().union(*(self.qualifier_key.get(key, set()) for key in keys)))
            candidates = set.intersection(*pools)
            candidates = {sid for sid in candidates if all(self._condition(self.statements[sid], condition) for condition in hypothesis)}
        relation_ids = set().union(*(self.relation[relation_id] for relation_id in relations))
        if hypothesis:
            candidates &= relation_ids
        else:
            candidates = relation_ids
        if direction == "forward":
            candidates &= self.head[entity]
        elif direction == "reverse":
            candidates &= self.tail[entity]
        else:
            candidates &= self.head[entity] | self.tail[entity]
        for sid in sorted(candidates):
            statement = self.statements[sid]
            if statement.head == entity and direction in {"forward", "both"}:
                yield statement, statement.tail
            if statement.tail == entity and direction in {"reverse", "both"}:
                yield statement, statement.head
