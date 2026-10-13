"""Read-only research workspace API for the React client. No credentials exposed."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.datasets import DATASETS
from app.services.memory import get_memory
from app.services.bert_embedder import get_bert_embedder
from app.services.detector import Exemplar, build_detection_prompt
from app.services.research_data import text_hash

router = APIRouter(prefix="/api/workspace", tags=["research workspace"])


def spec_for(dataset):
    if dataset not in DATASETS:
        raise HTTPException(404, "Unknown dataset")
    return DATASETS[dataset]


@lru_cache(maxsize=8)
def scan_dataset(dataset, directory, fingerprint):
    spec = spec_for(dataset)
    base = Path(directory)/spec.data_subdir
    splits, seen_train, overlap = {}, set(), 0
    for split in ["train", "test"]:
        path = base/f"{spec.file_prefix}_{split}_data.jsonl"
        count, empty, invalid, duplicates = 0, 0, 0, 0
        categories, labels, seen, preview = Counter(), Counter(), set(), []
        if not path.exists():
            splits[split] = {"count":0,"available":False,"preview":[]}
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = record.get("text", "")
                empty += not isinstance(text,str) or not text.strip()
                invalid += type(record.get("label")) is not int or record.get("label") not in (0,1)
                h = text_hash(str(text))
                duplicates += h in seen
                seen.add(h)
                if split=="test":
                    overlap += h in seen_train
                categories[str(record.get("original_label","Unknown"))] += 1
                labels[str(record.get("label"))] += 1
                if count < 50:
                    preview.append({"id":f"{split}_{count:05d}","text":text,"label":record.get("label"),"category":record.get("original_label"),"split":split})
                count += 1
        if split=="train":
            seen_train = seen
        splits[split] = {"count":count,"available":True,"normal":labels.get("0",0),"anomaly":labels.get("1",0),
                         "empty_text":empty,"invalid_labels":invalid,"duplicates":duplicates,"categories":dict(categories),"preview":preview}
    vector_file = base/f"{spec.file_prefix}_train_data_bert_base_uncased_feature.npy"
    shape = np.load(vector_file,mmap_mode="r",allow_pickle=False).shape if vector_file.exists() else None
    train = splits["train"]
    valid = bool(train.get("available") and train["count"] and not train["empty_text"] and not train["invalid_labels"] and train["anomaly"]==0)
    ready = valid and shape is not None and len(shape)==2 and shape[0]==train["count"]
    return {"id":dataset,"name":spec.display_name or dataset,"source":"NLP-ADBench", "splits":splits,
            "validation":{"normal_train_only":valid,"exact_overlap":overlap,"near_duplicates":"Not computed", "vector_rows_aligned":ready},
            "index":{"ready":ready,"provider":"NumPy exact cosine","embedding":settings.bert_model,"dimensions":shape[1] if shape else None,
                     "count":shape[0] if ready else 0,"split":"train","collection":f"{dataset}/normal_train",
                     "version":hashlib.sha256(repr(fingerprint).encode()).hexdigest()[:12],"created_at":None,
                     "source_modified_at":datetime.fromtimestamp(vector_file.stat().st_mtime,timezone.utc).isoformat() if vector_file.exists() else None}}


@router.get("/datasets")
def datasets():
    output = []
    for key,spec in DATASETS.items():
        directory = Path(settings.data_dir)/spec.data_subdir
        fingerprint = tuple((p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in sorted(directory.glob(f"{spec.file_prefix}*")))
        try:
            output.append(scan_dataset(key,str(settings.data_dir),fingerprint))
        except (ValueError,OSError) as exc:
            raise HTTPException(422,f"Dataset validation failed: {exc}") from exc
    return {"datasets":output,"configuration":{"llm":settings.llm_model_name,"provider":settings.llm_provider,
        "embedding":settings.bert_model,"version":settings.app_version},
        "capabilities":{"upload":False,"build_index":False,"retrieval":True,"detection":True,"model_selection":True}}


class SearchRequest(BaseModel):
    dataset: str = "n24_news"
    text: str = Field(min_length=1,max_length=8000)
    k: int = Field(3,ge=1,le=10)
    setting: Literal["normal_only","normal_anomaly"] = "normal_only"


@router.post("/retrieval")
def retrieval(request: SearchRequest):
    spec = spec_for(request.dataset)
    if not request.text.strip():
        raise HTTPException(422,"Text cannot be blank")
    try:
        neighbours = get_memory(request.dataset).search(get_bert_embedder().embed(request.text),request.k,exclude_text=request.text)
    except (FileNotFoundError,ValueError) as exc:
        raise HTTPException(503,str(exc)) from exc
    return {"references":[{"number":i,"doc_id":n.doc_id,"text":n.text,"category":n.category,"distance":n.distance,"split":"train","dataset":request.dataset} for i,n in enumerate(neighbours,1)],
            "prompt":build_detection_prompt(request.text,spec.normal_categories,spec.anomaly_category,request.setting,
                [Exemplar(n.text,n.category,doc_id=n.doc_id) for n in neighbours],include_reference_categories=False),
            "query_excluded":True,"reference_labels_in_prompt":False}


def run_files():
    return {p.parent.name:p for p in Path(settings.runs_dir).glob("*/summary.json")}


@router.get("/experiments")
def experiments():
    items=[]
    for name,path in sorted(run_files().items(),key=lambda kv:kv[1].stat().st_mtime,reverse=True):
        data=json.loads(path.read_text(encoding="utf-8"))
        if data.get("command") not in {"baselines","detect"}:
            continue
        manifest=path.parent/"manifest.json"
        config=json.loads(manifest.read_text(encoding="utf-8")).get("config",{}) if manifest.exists() else {}
        items.append({"id":name,"dataset":data["dataset"],"embedding":data.get("embedding"),"command":data["command"],
                      "n_train":data.get("n_train"),"n_test":data.get("n_test"),"seed":data.get("seed"),"results":data["results"],
                      "setting":config.get("setting"),"llm":config.get("llm"),"updated_at":datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                      "split_audit":data.get("split_audit"),"legacy_reference_categories":data["command"]=="detect"})
    return {"experiments":items}


@router.get("/experiments/{run_id}/predictions/{variant}")
def predictions(run_id:str,variant:str):
    if run_id not in run_files():
        raise HTTPException(404,"Unknown experiment")
    folder=run_files()[run_id].parent
    files={p.stem.removeprefix("predictions_"):p for p in folder.glob("predictions_*.json")}
    if variant not in files:
        raise HTTPException(404,"Unknown experiment variant")
    records=json.loads(files[variant].read_text(encoding="utf-8"))
    return {"predictions":[{"id":str(r["row"]),"label":r.get("label"),"score":r.get("score"),"error":r.get("error"),"pending":r.get("pending",False)} for r in records]}
