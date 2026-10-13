# RAG verification — 13 October 2026

Model: `nvidia/nemotron-3-super-120b-a12b:free`. Three live comparisons, same model and query for each pair, normal-plus-anomaly category setting, K=3.

| Sample | Without RAG | With RAG | RAG citations |
|---|---:|---:|---|
| food | 0.95 | 0.9 | [2] |
| science | 0.1 | 0.1 | [1] |
| borderline | 0.1 | 0.1 | [1] |

All six calls succeeded. Each RAG prompt contained the three actual retrieved text excerpts and document IDs. None of those excerpts appeared in the corresponding baseline prompt. All three pairs had different explanations. The frontend binds its two cards to separate response fields; no copied response was found.

Equal scores and classifications are valid outcomes, especially when category definitions already make a query easy to classify. A previous recipe run gave 0.95 for both; the rerun gave 0.95 and 0.90. Model/provider variation means this difference alone is not evidence of improvement.

Retrieval quality still needs evaluation: the recipe query retrieved a fashion snippet and a Health-labeled smoothie reference, while the astronomy query retrieved an archaeology reference. Valid citation numbers only prove that an existing reference was cited, not that it logically supports the decision. The current BERT CLS nearest-neighbor retrieval and overlapping dataset topics can supply weak context.

These three manual samples verify execution, not accuracy. A labeled held-out evaluation is required to compare AUROC, average precision, and threshold metrics. Raw requests' returned prompts, excerpts and responses are in runs/rag_audit/{food,science,borderline}.json.
