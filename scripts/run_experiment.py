#!/usr/bin/env python3
"""Reproducible anomaly detection and model selection on NLP-ADBench.

Run --help for commands. LLM commands require --allow-api; profiles/baselines are local.
Use a separate output directory for every configuration. Existing successful calls resume.
"""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from threadpoolctl import threadpool_limits
import numpy as np
from sklearn.model_selection import train_test_split
from app.core.config import settings
from app.core.datasets import DATASETS
from app.services.baselines import score_baseline
from app.services.detector import Exemplar, build_detection_prompt, detect
from app.services.evaluation import metrics, selection_metrics
from app.services.llm_client import get_llm_client
from app.services.model_selection import CARDS, CONDITIONS, build_selection_prompt, parse_selection
from app.services.profiling import profile_training
from app.services.research_data import eligible_test_rows, load_split, normalize, sha256


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def choose_rows(rows, labels, count, seed):
    if not count or count >= len(rows):
        return rows
    if count < 4:
        raise ValueError("Use at least four evaluation rows")
    selected, _ = train_test_split(rows, train_size=count, random_state=seed, stratify=labels[rows])
    return np.sort(selected)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["profile", "baselines", "detect", "select"])
    p.add_argument("--dataset", default="n24_news", choices=list(DATASETS))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--embedding", choices=["bert", "openai"], default="bert")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=2000, help="0 = full normal training split")
    p.add_argument("--max-test", type=int, default=1000, help="0 = full eligible test split")
    p.add_argument("--profile-samples", type=int, default=1000)
    p.add_argument("--models", nargs="+", choices=[*CARDS,"knn"], default=["iforest", "lof", "knn"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--ks", nargs="+", type=int, default=[0,1,3,5])
    p.add_argument("--setting", choices=["normal_only", "normal_anomaly"], default="normal_only")
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--allow-api", action="store_true", help="Explicitly enable configured LLM calls")
    p.add_argument("--max-calls", type=int, default=20, help="Hard cap on logical LLM calls in this invocation")
    p.add_argument("--candidate-results", help="Optional baseline summary to score recommendations after selection")
    args = p.parse_args(argv)
    if min(args.max_train, args.max_test, args.max_calls, args.seed) < 0 or args.epochs < 1 or not all(0 <= k <= 10 for k in args.ks):
        p.error("Invalid sample size, call limit, epoch count or k")
    if args.command in {"detect", "select"} and not args.allow_api:
        p.error("Use --allow-api to run the configured LLM, after reviewing the call budget")
    if args.command in {"detect", "select"} and settings.llm_provider == "stub":
        p.error("Configure a real LLM provider; stub outputs are never evaluated")
    spec = DATASETS[args.dataset]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = load_split(Path(args.data_dir)/spec.data_subdir, spec.file_prefix, "train", args.embedding)
    config = vars(args).copy()
    config.pop("max_calls")
    config.pop("allow_api")
    config["llm"] = {"provider": settings.llm_provider, "model": settings.llm_model_name,
                     "temperature": settings.llm_temperature, "max_tokens": settings.llm_max_tokens}
    config["source_hashes"] = {str(f.relative_to(ROOT)): sha256(f) for f in sorted((ROOT/"backend/app/services").glob("*.py"))}
    config["source_hashes"]["scripts/run_experiment.py"] = sha256(Path(__file__))
    manifest = {"config": config, "files": {str(f):sha256(f) for f in [train.text_path, train.vector_path]},
                "python": platform.python_version(), "packages": {n:importlib.metadata.version(n) for n in ["numpy", "scipy", "scikit-learn"]}}
    test = None
    if args.command in {"baselines", "detect"}:
        test = load_split(Path(args.data_dir)/spec.data_subdir, spec.file_prefix, "test", args.embedding)
        manifest["files"].update({str(f):sha256(f) for f in [test.text_path,test.vector_path]})
    manifest_file = out/"manifest.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Output directory belongs to a different configuration/data/code; use a new --out")
    save(manifest_file, manifest)
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(len(train.records), min(args.max_train or len(train.records),len(train.records)), replace=False))
    if len(rows)<3:
        raise ValueError("Need at least three training records")
    x = normalize(train.vectors[rows])
    records = [train.records[i] for i in rows]
    profile = profile_training(records, x, seed=args.seed, max_samples=args.profile_samples)
    profile["benchmark_train_rows"] = rows.tolist()
    save(out/"profile.json",profile)
    if args.command == "profile":
        print(f"Saved training profile: {out/'profile.json'}")
        return
    summary = {"schema_version": 1, "command": args.command, "dataset": args.dataset, "embedding": args.embedding,
               "seed": args.seed, "n_train":len(rows), "results": []}
    summary["train_signature"] = hashlib.sha256(json.dumps({"vectors":sha256(train.vector_path),
        "texts":sha256(train.text_path),"rows":rows.tolist(),"preprocessing":"unit_l2"},sort_keys=True).encode()).hexdigest()
    if test:
        eligible, audit = eligible_test_rows(train,test)
        test_rows = choose_rows(eligible,test.labels,args.max_test,args.seed)
        if not len(test_rows):
            raise ValueError("No eligible test records")
        z = normalize(test.vectors[test_rows])
        labels = test.labels[test_rows]
        save(out/"split.json", {"train_rows":rows.tolist(),"test_rows":test_rows.tolist(),**audit})
        summary.update(n_test=len(test_rows), split_audit=audit)
    calls = 0
    if args.command == "baselines":
        for name in args.models:
            started = time.perf_counter()
            try:
                scores, params = score_baseline(name,x,z,args.seed,args.epochs)
                if len(scores)!=len(labels) or not np.isfinite(scores).all():
                    raise ValueError("Detector produced invalid scores")
                result = {"model":name,"metrics":metrics(labels,scores,seed=args.seed), "parameters":params,
                          "seconds":time.perf_counter()-started}
                save(out/f"predictions_{name}.json",[{"row":int(i),"label":int(y),"score":float(s)} for i,y,s in zip(test_rows,labels,scores)])
            except Exception as exc:
                result = {"model":name,"error":f"{type(exc).__name__}: {exc}"}
            summary["results"].append(result)
            save(out/"summary.json",summary)
            print(json.dumps(result),flush=True)
    elif args.command == "select":
        candidates = [m for m in args.models if m in CARDS]
        if not candidates:
            raise ValueError("Selection requires candidate model cards")
        llm = get_llm_client()
        for condition in args.conditions:
            prompt, features = build_selection_prompt(profile,args.dataset,candidates,condition,args.seed)
            (out/f"prompt_{condition}.txt").write_text(prompt,encoding="utf-8")
            path = out/f"selection_{condition}.json"
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
            elif calls < args.max_calls:
                calls += 1
                started = time.perf_counter()
                try:
                    response = llm.generate(prompt,max_tokens=max(2048,settings.llm_max_tokens))
                    result = {"condition":condition,"model":response.model,"raw_text":response.text,
                              "usage":response.raw.get("usage"),"seconds":time.perf_counter()-started}
                    try:
                        result.update(parse_selection(response.text,candidates,features))
                    except (ValueError,TypeError,AttributeError) as exc:
                        result["error"] = str(exc)
                except Exception as exc:
                    result = {"condition":condition,"error":str(exc)}
                save(path,result)
            else:
                result = {"condition":condition,"pending":True}
            # Read held-out metrics only AFTER recommendation is generated and saved.
            if args.candidate_results and result.get("ranking"):
                measured = json.loads(Path(args.candidate_results).read_text(encoding="utf-8"))
                if measured.get("train_signature") != summary["train_signature"]:
                    raise ValueError("Candidate results must use matching dataset, embedding, seed and train size")
                aucs = {r["model"]:r["metrics"]["auroc"] for r in measured["results"] if r.get("metrics") and r["model"] in candidates}
                result["evaluation"] = selection_metrics(result["ranking"],aucs)
            summary["results"].append(result)
            save(out/"summary.json",summary)
            print(f"{condition}: {result.get('selected_model',result.get('error','pending'))}",flush=True)
    else:
        llm = get_llm_client()
        # Interleave variants by test row so a small call budget yields paired data.
        by_k = {k: [] for k in sorted(set(args.ks))}
        for row, vector, label in zip(test_rows,z,labels):
            for k, predictions in by_k.items():
                path = out/f"calls/k{k}_{int(row)}.json"
                if path.exists():
                    item = json.loads(path.read_text(encoding="utf-8"))
                elif calls < args.max_calls:
                    neighbours = []
                    if k:
                        similarity = x@vector
                        nearest = np.argsort(-similarity,kind="stable")[:k]
                        neighbours = [Exemplar(text=records[j]["text"],category=str(records[j]["original_label"]),doc_id=f"trn_{int(rows[j]):05d}") for j in nearest]
                    prompt = build_detection_prompt(test.records[row]["text"], spec.normal_categories,spec.anomaly_category,args.setting,neighbours)
                    calls += 1
                    started = time.perf_counter()
                    result = detect(llm,prompt,n_exemplars=len(neighbours),max_tokens=max(1024,settings.llm_max_tokens))
                    item = {**dataclasses.asdict(result), "row":int(row),"label":int(label),"k":k,
                            "retrieved_ids":[n.doc_id for n in neighbours],"seconds":time.perf_counter()-started}
                    save(path,item)
                else:
                    item = {"row":int(row),"label":int(label),"score":None,"pending":True}
                predictions.append(item)
        for k, predictions in by_k.items():
            result = {"k":k,"setting":args.setting,"metrics":metrics([p['label'] for p in predictions],[p['score'] for p in predictions],seed=args.seed),
                      "pending":sum(bool(p.get('pending')) for p in predictions),
                      "api_errors":sum(bool(p.get('error')) for p in predictions)}
            summary["results"].append(result)
            save(out/f"predictions_k{k}.json",predictions)
            save(out/"summary.json",summary)
            print(json.dumps(result),flush=True)
        # Baseline/retrieval comparison uses the same successfully scored rows.
        completed = {r["k"]:json.loads((out/f"predictions_k{r['k']}.json").read_text(encoding="utf-8")) for r in summary["results"]}
        common = set.intersection(*[{p["row"] for p in ps if p.get("score") is not None} for ps in completed.values()])
        summary["common_valid_rows"] = sorted(common)
        summary["paired_metrics"] = {str(k):metrics([p["label"] for p in ps if p["row"] in common],[p["score"] for p in ps if p["row"] in common],seed=args.seed) for k,ps in completed.items()}
    summary["logical_llm_calls_this_invocation"] = calls
    save(out/"summary.json",summary)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
