import json
from collections import defaultdict

from graph import norm


class HoG:
    def __init__(self, graph, llm, encoders, retrieval_k=10, pruning_k=3):
        self.graph = graph
        self.llm = llm
        self.encoders = encoders
        self.retrieval_k = retrieval_k
        self.pruning_k = pruning_k

    def identify(self, question):
        extracted = self.llm.json(
            "Extract every explicitly mentioned named entity and literal qualifier value from the question. Do not infer intermediate or answer entities.",
            {"question": question, "format": {"entity_mentions": ["exact surface form"]}},
        )
        mentions = extracted.get("entity_mentions", [])
        if not isinstance(mentions, list):
            raise ValueError("entity_mentions must be a list")
        candidates = []
        for mention in mentions:
            for entity in self.encoders.link(self.graph, str(mention)):
                evidence = self.encoders.top_statements(self.graph, question, self.graph.incident(entity), self.retrieval_k)
                candidates.append({
                    "mention": str(mention),
                    "id": entity,
                    "label": self.graph.label(entity),
                    "evidence": [self.graph.serialized(statement) for statement in evidence],
                })
        decision = self.llm.json(
            "Classify the graph-linked entity candidates as topic entities or qualifier values using the question and the supplied complete statement evidence. Select only candidate IDs, retain all independent source branches, and never use an answer or hidden gold path.",
            {"question": question, "candidates": candidates, "format": {"topic_ids": ["candidate ID"], "qualifier_value_ids": ["candidate ID"]}},
        )
        available = {candidate["id"] for candidate in candidates}
        topics = [str(value) for value in decision.get("topic_ids", [])]
        if not topics or any(value not in available for value in topics):
            raise ValueError("Topic identification returned missing or invalid IDs")
        return list(dict.fromkeys(topics)), candidates, decision

    def decompose(self, question, topics, role_decision):
        relation_inventory = [{"id": key, "label": value} for key, value in self.graph.relations.items()]
        result = self.llm.json(
            "Decompose the question into ordered dependent subquestions and a qualifier-hypothesis reasoning tree plan. Each relation step must specify a graph relation ID, direction forward or reverse, input reference source:N or a prior step ID, and one or more alternative qualifier combinations. An empty combination is allowed when appropriate. For multiple sources, expand each source branch and add a join step that requires their intermediate answers to be the same; add continuation steps after the join if needed. Every qualifier condition expressed in the question must appear in required_constraints with its owning relation step. Never use the gold answer or reasoning path.",
            {
                "question": question,
                "topics": [{"ref": f"source:{i}", "id": entity, "label": self.graph.label(entity)} for i, entity in enumerate(topics)],
                "qualifier_value_ids": role_decision.get("qualifier_value_ids", []),
                "relations": relation_inventory,
                "format": {
                    "steps": [{"id": "g1", "kind": "relation", "question": "subquestion", "input": "source:0", "relation": "relation ID", "direction": "forward", "qualifier_hypotheses": [[{"key": "qualifier relation ID", "value": "value ID or literal"}], []]}, {"id": "join1", "kind": "join", "question": "shared intermediate entity", "inputs": ["g1", "g2"]}],
                    "required_constraints": [{"step": "g1", "key": "qualifier relation ID", "value": "value ID or literal"}],
                    "answer_ref": "last answer-producing step ID",
                },
            },
        )
        steps = result.get("steps", [])
        if not isinstance(steps, list) or not steps:
            raise ValueError("No reasoning steps returned")
        seen = {f"source:{i}" for i in range(len(topics))}
        for step in steps:
            sid = str(step.get("id", ""))
            if not sid or sid in seen:
                raise ValueError(f"Duplicate or missing step ID: {sid}")
            kind = step.get("kind", "relation")
            refs = step.get("inputs", []) if kind == "join" else [step.get("input")]
            if not refs or any(ref not in seen for ref in refs):
                raise ValueError(f"Invalid input references in {sid}")
            if kind == "relation":
                if not self.graph.relation_candidates(str(step.get("relation", ""))):
                    raise ValueError(f"Unknown relation in {sid}")
                if step.get("direction") not in {"forward", "reverse", "both"}:
                    raise ValueError(f"Invalid direction in {sid}")
                hypotheses = step.get("qualifier_hypotheses")
                if not isinstance(hypotheses, list) or not hypotheses or any(not isinstance(item, list) for item in hypotheses):
                    raise ValueError(f"Invalid qualifier hypotheses in {sid}")
            elif kind != "join":
                raise ValueError(f"Unknown step kind: {kind}")
            seen.add(sid)
        if result.get("answer_ref") not in seen:
            raise ValueError("Invalid answer reference")
        constraints = result.get("required_constraints", [])
        if not isinstance(constraints, list) or any(item.get("step") not in seen for item in constraints):
            raise ValueError("Invalid required constraints")
        return result

    def explore(self, plan, topics):
        states = [{"bindings": {f"source:{i}": entity for i, entity in enumerate(topics)}, "evidence": {}, "trace": []}]
        counts = []
        for step in plan["steps"]:
            expanded = []
            sid = step["id"]
            for state in states:
                if step.get("kind", "relation") == "join":
                    values = [state["bindings"][ref] for ref in step["inputs"]]
                    if len(set(values)) == 1:
                        new = {"bindings": {**state["bindings"], sid: values[0]}, "evidence": dict(state["evidence"]), "trace": state["trace"] + [{"step": sid, "kind": "join", "entity": values[0]}]}
                        expanded.append(new)
                    continue
                entity = state["bindings"][step["input"]]
                for hypothesis in step["qualifier_hypotheses"]:
                    for statement, answer in self.graph.matches(step["relation"], entity, step["direction"], hypothesis):
                        new = {
                            "bindings": {**state["bindings"], sid: answer},
                            "evidence": {**state["evidence"], sid: statement.sid},
                            "trace": state["trace"] + [{"step": sid, "kind": "relation", "input": entity, "entity": answer, "statement_id": statement.sid, "hypothesis": hypothesis}],
                        }
                        expanded.append(new)
            counts.append({"step": sid, "candidates": len(expanded)})
            states = expanded
            if not states:
                break
        return states, counts

    def filter_complete(self, plan, states):
        required = plan.get("required_constraints", [])
        valid = []
        for state in states:
            if any(step["id"] not in state["bindings"] for step in plan["steps"]):
                continue
            if all(constraint["step"] in state["evidence"] and self.graph._condition(self.graph.statements[state["evidence"][constraint["step"]]], constraint) for constraint in required):
                valid.append(state)
        return valid

    def _branch_text(self, state):
        return [{"step": entry["step"], "entity": self.graph.label(entry["entity"]), "statement_id": entry.get("statement_id"), "statement": self.graph.serialized(self.graph.statements[entry["statement_id"]]) if "statement_id" in entry else None} for entry in state["trace"]]

    def prune(self, question, plan, states):
        steps = plan["steps"]
        for depth in range(len(steps) - 1, -1, -1):
            groups = defaultdict(list)
            for state in states:
                prefix = tuple((item["step"], item["entity"], item.get("statement_id")) for item in state["trace"][:depth])
                groups[prefix].append(state)
            kept = []
            for group in groups.values():
                if len(group) <= self.pruning_k:
                    kept.extend(group)
                    continue
                choices = [{"candidate": i, "evidence": self._branch_text(state)} for i, state in enumerate(group)]
                selected = self.llm.json(
                    "Rank complete, qualifier-valid reasoning branches by consistency with the original question and their statement-level evidence. Select at most the requested number of candidate indices. Do not invent facts or select an unlisted candidate.",
                    {"question": question, "subquestion": question if depth == 0 else steps[depth - 1]["question"], "candidates": choices, "keep": self.pruning_k, "format": {"selected": [0]}},
                ).get("selected", [])
                if not isinstance(selected, list) or not selected or len(set(selected)) > self.pruning_k or any(not isinstance(i, int) or i < 0 or i >= len(group) for i in selected):
                    raise ValueError("Invalid pruning selection")
                kept.extend(group[i] for i in dict.fromkeys(selected))
            states = kept
        return states

    def answer(self, question, plan, states):
        if not states:
            return {"answers": [], "supporting_statement_ids": []}
        candidates = [{"candidate": i, "answer_entity": self.graph.label(state["bindings"][plan["answer_ref"]]), "evidence": self._branch_text(state)} for i, state in enumerate(states)]
        result = self.llm.json(
            "Answer using only the provided retained, qualifier-valid HKG branches. Return all and only supported answer entity labels and their supporting statement IDs. An unsupported answer must be omitted.",
            {"question": question, "candidates": candidates, "format": {"answers": ["entity label"], "supporting_statement_ids": [0]}},
        )
        possible = {norm(item["answer_entity"]): item["answer_entity"] for item in candidates}
        answers = [possible[norm(value)] for value in result.get("answers", []) if norm(value) in possible]
        support = {entry["statement_id"] for state in states for entry in state["trace"] if "statement_id" in entry}
        cited = [int(value) for value in result.get("supporting_statement_ids", []) if str(value).isdigit() and int(value) in support]
        return {"answers": list(dict.fromkeys(answers)), "supporting_statement_ids": sorted(set(cited))}

    def solve(self, question):
        topics, candidates, roles = self.identify(question)
        plan = self.decompose(question, topics, roles)
        expanded, counts = self.explore(plan, topics)
        valid = self.filter_complete(plan, expanded)
        kept = self.prune(question, plan, valid)
        result = self.answer(question, plan, kept)
        return {"topics": [{"id": value, "label": self.graph.label(value)} for value in topics], "plan": plan, "candidate_counts": counts, "expanded_branches": len(expanded), "valid_branches": len(valid), "retained_branches": len(kept), "answer": result, "retained_evidence": [self._branch_text(state) for state in kept]}
