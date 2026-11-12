# HoG

`data/` contains four local evaluation datasets: `hywiki` (2,022 questions), `hywd` (1,259), `kqapro` (73), and `lcquad2` (402). Each directory contains the complete graph used for that local evaluation, entity and relation label maps, and `qa.json`. Each QA row has only `index`, `Question`, `Answer`, `Source Entity`, and `Reasoning Path`. The local KQA Pro and LC-QuAD 2.0 QA files do not provide annotated reasoning paths, so their `Reasoning Path` values are `null`. The KQA Pro and LC-QuAD graphs are the local evaluation subgraphs described in the paper.

`code/` implements the paper's topic identification, qualifier-hypothesis exploration, constraint filtering, bottom-up pruning, and evidence-grounded answer generation. It uses frozen LLMs, MiniLM linking, SBERT statement retrieval (`K=10`), and at most three branches per node (`Kp=3`). The run reads only `Question` from each QA row; `Answer`, `Source Entity`, and `Reasoning Path` are never supplied to the model.

Install dependencies and provide an OpenAI-compatible API key and optional endpoint:

```bash
cd HoG
export HOG_API_KEY=your_key
export HOG_BASE_URL=https://your-provider.example/v1
python code/run.py --dataset hywiki --model gemini-2.5-flash
```
