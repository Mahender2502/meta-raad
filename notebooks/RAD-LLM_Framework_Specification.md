# RAD-LLM: Retrieval-Augmented Deliberative LLM-based Anomaly Detection

## Complete Framework Specification

---

## 1. Framework Overview

RAD-LLM extends the AD-LLM baseline with three novel components that address its core limitations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RAD-LLM FRAMEWORK                                   │
│                                                                             │
│   ┌─────────────┐   ┌──────────────────┐   ┌────────────────────────┐      │
│   │  Component 1 │   │   Component 2     │   │    Component 3          │      │
│   │  RAG-AD      │   │   Self-Consistency │   │    Explanation          │      │
│   │  Module      │   │   Scoring Module   │   │    Evaluation Module    │      │
│   └─────────────┘   └──────────────────┘   └────────────────────────┘      │
│                                                                             │
│   Retrieval of        Multi-path LLM         Automated assessment           │
│   similar normal      querying with          of explanation quality          │
│   samples as          aggregated scores      (faithfulness, specificity,     │
│   few-shot context    + uncertainty          informativeness)                │
│                       estimation                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**What AD-LLM does**: category name + test sample → single LLM call → score
**What RAD-LLM does**: retrieved exemplars + category name + test sample → multiple LLM calls → aggregated score + confidence + explanation quality score

---

## 2. Detailed Architecture

### 2.1 Component 1: Retrieval-Augmented Anomaly Detection (RAG-AD)

**Problem it solves**: AD-LLM gives the LLM only abstract category names (e.g., "Sports"). The LLM must rely entirely on its pre-trained knowledge to decide what "Sports" text looks like. This is fragile — the LLM's internal concept of "Sports" may not match the dataset's distribution.

**Solution**: At inference time, retrieve the k most similar normal training samples and include them as concrete examples in the prompt.

```
                    OFFLINE (one-time setup)
                    ========================

┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Normal       │     │ Embedding       │     │ FAISS Vector     │
│ Training     │────>│ Model           │────>│ Index             │
│ Samples      │     │ (sentence-      │     │ (stores all       │
│ D_train      │     │  transformers)  │     │  normal sample    │
│              │     │                 │     │  embeddings)      │
└──────────────┘     └─────────────────┘     └──────────────────┘


                    ONLINE (per test sample)
                    ========================

┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Test Sample  │     │ Embed test      │     │ FAISS Search     │
│ x_i          │────>│ sample          │────>│ top-k nearest    │
│              │     │                 │     │ neighbors        │
└──────────────┘     └─────────────────┘     └───────┬──────────┘
                                                     │
                                                     v
                                             ┌──────────────────┐
                                             │ Retrieved normal │
                                             │ exemplars        │
                                             │ {e_1, e_2, ...,  │
                                             │  e_k}            │
                                             └───────┬──────────┘
                                                     │
                                          ┌──────────┘
                                          v
                                 ┌─────────────────┐
                                 │ RAG-Enhanced     │
                                 │ Prompt           │
                                 │ (categories +    │
                                 │  exemplars +     │
                                 │  test sample)    │
                                 └─────────────────┘
```

**RAG-Enhanced Prompt Template** (new — extends Table A9/A10):

```
You are an intelligent and professional assistant that detects anomalies
in text data.

## Task:
- Determine whether the given text sample is an anomaly...

## Categories:
- **Sports**
- **Business**
- **Sci/Tech**

## Reference Examples of Normal Data:
Below are real examples from each normal category to help you understand
what normal data looks like in this dataset:

### Sports Examples:
1. "Tiger Woods wins the Masters tournament in dramatic fashion..."
2. "The Lakers defeated the Celtics 108-102 in overtime..."

### Business Examples:
1. "Apple Inc reported Q3 earnings of $1.20 per share..."
2. "The Federal Reserve held interest rates steady at..."

### Sci/Tech Examples:
1. "NASA's Perseverance rover successfully collected its first..."
2. "Google announced a breakthrough in quantum computing..."

## Rules:
[same CoT rules as AD-LLM]

Text sample:
"{text}"

Response in JSON format:
```

**Key design decisions for RAG-AD**:
- k = 3 exemplars per normal category (ablation study: test k = 1, 2, 3, 5)
- Embedding model: `all-MiniLM-L6-v2` (fast, 384-dim) or `text-embedding-3-small` (OpenAI)
- Retrieval strategy: retrieve per-category (not global) to ensure balanced representation
- Similarity metric: cosine similarity


### 2.2 Component 2: Self-Consistency Scoring

**Problem it solves**: AD-LLM queries the LLM once at temperature=0. A single deterministic pass provides no uncertainty information. The LLM might output 0.85 for a sample it's genuinely unsure about.

**Solution**: Query the LLM m times with temperature > 0 and slightly varied prompts. Aggregate the scores and use variance as a confidence measure.

```
                         Test Sample x_i
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                    v         v         v
              ┌──────────┐ ┌──────────┐ ┌──────────┐    ... m paths
              │ Prompt   │ │ Prompt   │ │ Prompt   │
              │ Variant 1│ │ Variant 2│ │ Variant 3│
              │ (temp=0.3│ │ (temp=0.5│ │ (temp=0.7│
              │  seed=42)│ │  seed=43)│ │  seed=44)│
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            │            │
                   v            v            v
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │ (r₁, s₁) │ │ (r₂, s₂) │ │ (r₃, s₃) │
              │ reason,  │ │ reason,  │ │ reason,  │
              │ score    │ │ score    │ │ score    │
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            │            │
                   └────────────┼────────────┘
                                │
                                v
                    ┌───────────────────────┐
                    │     AGGREGATION       │
                    │                       │
                    │ s_final = mean(s₁..sₘ)│
                    │                       │
                    │ confidence =          │
                    │   1 - std(s₁..sₘ)    │
                    │                       │
                    │ High variance =       │
                    │   uncertain sample    │
                    │   → flag for review   │
                    └───────────────────────┘
```

**Prompt variation strategies** (to ensure diverse reasoning paths):

| Variant | Change | Purpose |
|---------|--------|---------|
| 1 | Base prompt, temp=0.3 | Near-deterministic baseline |
| 2 | Reorder categories, temp=0.5 | Test if category order biases result |
| 3 | Rephrase anomaly definition, temp=0.5 | Test sensitivity to wording |
| 4 | Add "think carefully", temp=0.7 | Encourage deeper reasoning |
| 5 | Reverse CoT order (score facets), temp=0.7 | Different reasoning decomposition |

**Aggregation methods to compare** (ablation):
- Mean: `s_final = mean(s_1, ..., s_m)`
- Median: `s_final = median(s_1, ..., s_m)` (robust to outlier paths)
- Weighted mean: weight each score by the coherence of its explanation

**Uncertainty quantification**:
- `uncertainty = std(s_1, ..., s_m)`
- Samples with uncertainty > threshold → "uncertain" bucket
- This enables a **selective prediction** mode: only classify confident samples automatically, flag uncertain ones for human review


### 2.3 Component 3: Explanation Quality Evaluation (LLM-as-a-Judge)

**Problem it solves**: AD-LLM generates explanations alongside scores but never evaluates them. In real-world AD (healthcare, finance, cybersecurity), a score alone is useless — stakeholders need to understand WHY something was flagged.

**Solution**: Use a separate LLM (or the same LLM with a different role) to evaluate each explanation on three dimensions.

```
              ┌──────────────────────────────┐
              │  Original Detection Output   │
              │  - test sample x_i           │
              │  - reason r                  │
              │  - anomaly_score s           │
              │  - categories used           │
              └──────────────┬───────────────┘
                             │
                             v
              ┌──────────────────────────────┐
              │  JUDGE LLM                   │
              │  (separate evaluation call)  │
              │                              │
              │  Evaluates on 3 dimensions:  │
              │                              │
              │  1. FAITHFULNESS (1-5)       │
              │     Does the explanation     │
              │     accurately reflect the   │
              │     content of the text?     │
              │     No hallucinated facts?   │
              │                              │
              │  2. SPECIFICITY (1-5)        │
              │     Does it reference        │
              │     specific details from    │
              │     the text, or just give   │
              │     generic reasoning?       │
              │                              │
              │  3. INFORMATIVENESS (1-5)    │
              │     Does it help a human     │
              │     understand the decision? │
              │     Could you act on it?     │
              └──────────────┬───────────────┘
                             │
                             v
              ┌──────────────────────────────┐
              │  Explanation Quality Score    │
              │  EQ = (F + S + I) / 15       │
              │  Range: 0.2 to 1.0           │
              └──────────────────────────────┘
```

**Judge Prompt Template**:

```
You are an expert evaluator assessing the quality of anomaly detection
explanations.

## Context:
- Text sample: "{text}"
- Detection result: anomaly_score = {score}
- Categories: Normal = {normal_cats}, Anomaly = {anomaly_cat}

## Explanation to evaluate:
"{reason}"

## Evaluate on three dimensions (score 1-5 each):

1. **Faithfulness**: Does the explanation accurately describe the text
   content? Are there any hallucinated or incorrect claims about what
   the text says?

2. **Specificity**: Does the explanation reference specific details,
   keywords, or themes from the text? Or is it generic (e.g., "this
   text does not fit the categories")?

3. **Informativeness**: Would a human reviewer understand WHY this
   sample was flagged/not flagged based on this explanation alone?

Response in JSON:
{"faithfulness": <1-5>, "specificity": <1-5>, "informativeness": <1-5>,
 "justification": "<brief justification for scores>"}
```

**Analysis we'll perform**:
- Correlation between explanation quality and detection accuracy
- Do better-explained predictions tend to be correct predictions?
- Comparison across LLMs: which model gives the best explanations?
- Comparison: does RAG context improve explanation quality?

---

## 3. Complete Pipeline (All Components Combined)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       RAD-LLM FULL PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────┘

PHASE 0: OFFLINE SETUP
══════════════════════
  Normal Training Data ──> Embed with sentence-transformers ──> FAISS Index


PHASE 1: RETRIEVAL (Component 1)
════════════════════════════════
  Test sample x_i ──> Embed ──> FAISS search ──> top-k exemplars per category


PHASE 2: MULTI-PATH DETECTION (Components 1 + 2)
═════════════════════════════════════════════════

  For each prompt variant j = 1..m:
    ┌────────────────────────────────────────────────────┐
    │  Construct RAG-enhanced prompt P_j:                │
    │    - System: role definition                       │
    │    - Categories: normal (+ anomaly if Setting 2)   │
    │    - Retrieved exemplars: {e_1, ..., e_k}          │
    │    - Prompt variant modifications                  │
    │    - CoT reasoning instructions                    │
    │    - Test sample x_i                               │
    │    - JSON output format                            │
    └────────────────────┬───────────────────────────────┘
                         │
                         v
    ┌────────────────────────────────────────────────────┐
    │  LLM inference: (r_j, s_j) = f_LLM(P_j)          │
    └────────────────────────────────────────────────────┘

  Aggregate:
    s_final = mean(s_1, ..., s_m)
    uncertainty = std(s_1, ..., s_m)
    best_reason = r_j where j = argmin |s_j - s_final|


PHASE 3: EXPLANATION EVALUATION (Component 3)
═════════════════════════════════════════════
  (x_i, best_reason, s_final) ──> Judge LLM ──> (faithfulness, specificity,
                                                  informativeness)


PHASE 4: FINAL OUTPUT
════════════════════
  For each test sample x_i:
    {
      "anomaly_score": s_final,        # aggregated score
      "uncertainty": uncertainty,       # confidence estimate
      "explanation": best_reason,       # most representative explanation
      "explanation_quality": {
        "faithfulness": F,
        "specificity": S,
        "informativeness": I
      },
      "flag_for_review": uncertainty > threshold
    }


PHASE 5: EVALUATION
═══════════════════
  - AUROC, AUPRC (detection quality — compare vs AD-LLM)
  - Uncertainty calibration plots (reliability diagrams)
  - Explanation quality distribution (histograms per LLM)
  - Selective prediction curves (accuracy vs coverage tradeoff)
```

---

## 4. Dataset Requirements

### 4.1 Primary Datasets (Same as AD-LLM — for direct comparison)

You MUST use the same 5 datasets to enable fair comparison with the baseline:

| Dataset | Source | Normal Categories | Anomaly Category | Train Size | Test Size | Anomaly % |
|---------|--------|-------------------|------------------|------------|-----------|-----------|
| AG News | News topic classification | Sports, Business, Sci/Tech | World | 66,098 | 32,109 | 11.77% |
| BBC News | BBC topic classification | Business, Politics, Sport, Tech | Entertainment | 1,206 | 579 | 10.71% |
| IMDB Reviews | Sentiment classification | Positive | Negative | 17,417 | 8,952 | 16.61% |
| N24 News | NYT news classification | 23 categories (Television, Your Money, etc.) | Food | 40,569 | 19,227 | 9.51% |
| SMS Spam | Spam detection | Non-spam (Ham) | Spam | 3,162 | 1,510 | 10.20% |

**Where to get them**: All 5 are sourced from the NLPADBench benchmark (Li et al., 2024c) and are available through the AD-LLM GitHub repository: https://github.com/USC-FORTIS/AD-LLM

**Dataset characteristics** (important for our adaptive experiments):

| Dataset | Avg Text Length | Max Length | Min Length | Std Dev | Type |
|---------|----------------|------------|------------|---------|------|
| AG News | 190 chars | 959 | 35 | 61.7 | Short news snippets |
| BBC News | 2,293 chars | 25,367 | 685 | 1,506.4 | Full articles |
| IMDB Reviews | 1,289 chars | 12,498 | 65 | 980.5 | Sentiment-heavy reviews |
| N24 News | 4,633 chars | 28,616 | 4 | 3,069.5 | Long-form news, 23 normal categories |
| SMS Spam | 78 chars | 790 | 4 | 60.8 | Very short messages |

### 4.2 Optional Extension Datasets (for broader evaluation — strengthens the paper)

If time permits, adding 1-2 extra datasets shows generalization:

| Dataset | Why Include It | Source |
|---------|---------------|--------|
| 20 Newsgroups | Classic NLP benchmark, 20 categories | scikit-learn built-in |
| Yelp Reviews | Different domain (restaurant reviews) | Hugging Face datasets |
| Enron Email | Real-world fraud/anomaly detection | Available online |

### 4.3 Data Splits and Protocol

Follow AD-LLM's exact protocol for fair comparison:

- **Training set**: Contains ONLY normal samples (used for building our FAISS index in RAG-AD)
- **Test set**: Mix of normal + anomaly samples with ground truth labels
- **No validation set in AD-LLM**: We will create one by holding out 10% of the test set for calibration experiments (report this clearly in the paper)
- **Seed**: 42 (same as AD-LLM)

---

## 5. Software Requirements

### 5.1 Python Environment

```
Python >= 3.11 (match AD-LLM)
```

### 5.2 Core Dependencies

```
# requirements.txt

# --- AD-LLM baseline dependencies ---
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
torch>=2.0.0
transformers>=4.35.0
accelerate>=0.24.0
pyod>=1.1.0                    # Traditional AD baselines

# --- RAD-LLM new dependencies ---

# LLM APIs
openai>=1.12.0                 # GPT-4o and text-embedding-3-small
# (DeepSeek uses the same openai SDK with different base_url)

# Retrieval / RAG
faiss-cpu>=1.7.4               # Vector similarity search index
# OR faiss-gpu>=1.7.4          # If GPU available
sentence-transformers>=2.2.0   # For embedding text samples

# Evaluation & Visualization
matplotlib>=3.7.0              # Plots for paper figures
seaborn>=0.12.0                # Statistical visualizations
pandas>=2.0.0                  # Data manipulation
tqdm>=4.65.0                   # Progress bars

# Utilities
jsonlines>=3.1.0               # For saving results line-by-line
python-dotenv>=1.0.0           # API key management
tenacity>=8.2.0                # Retry logic for API calls
```

### 5.3 Installation Commands

```bash
# Create conda environment (matching AD-LLM)
conda create -n rad-llm python=3.11 -y
conda activate rad-llm

# Install PyTorch (CUDA 12.1 — adjust for your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt

# For local Llama 3.1 inference
pip install vllm>=0.3.0  # Faster inference than raw transformers

# Download embedding model (one-time)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Download Llama 3.1 (if running locally)
# Requires Hugging Face access token with Llama license agreement
huggingface-cli login
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')"
```

---

## 6. Hardware Requirements

### 6.1 Minimum Setup (API-only, no local LLM)

| Component | Requirement |
|-----------|-------------|
| CPU | Any modern CPU (4+ cores) |
| RAM | 16 GB |
| GPU | NOT required (all inference via API) |
| Storage | 5 GB (datasets + results) |
| Internet | Required (for OpenAI / DeepSeek API calls) |

**Estimated API costs**:

| LLM | Cost per 1M input tokens | Cost per 1M output tokens | Est. cost for full experiment |
|-----|--------------------------|---------------------------|------------------------------|
| GPT-4o | $2.50 | $10.00 | ~$50-80 (5 datasets x 2 settings x m=5 paths) |
| GPT-4o-mini | $0.15 | $0.60 | ~$3-5 (budget alternative) |
| DeepSeek-V3 | $0.27 | $1.10 | ~$5-10 |

**Recommendation**: Use GPT-4o-mini for development/debugging, GPT-4o for final experiments.

### 6.2 Full Setup (with local Llama 3.1)

| Component | Requirement |
|-----------|-------------|
| CPU | 8+ cores |
| RAM | 32 GB |
| GPU | NVIDIA GPU with 24+ GB VRAM (RTX 3090/4090 or A100) |
| | Paper used RTX 6000 Ada (48 GB) |
| Storage | 30 GB (model weights + datasets + results) |

### 6.3 Recommended College Lab Setup

If your college has a GPU cluster or cloud credits:
- **Google Colab Pro+**: A100 GPU, sufficient for Llama 3.1 8B with quantization
- **AWS/GCP**: g5.xlarge (A10G 24GB) — ~$1/hour
- **Kaggle**: Free P100 GPU (16GB) — may need 4-bit quantization for Llama

---

## 7. API Requirements

### 7.1 OpenAI API

```
Required for: GPT-4o, GPT-4o-mini, text-embedding-3-small
Signup: https://platform.openai.com
Minimum credit: $10-20 for development, $50-100 for full experiments
```

**Environment setup**:
```bash
# .env file
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 7.2 DeepSeek API

```
Required for: DeepSeek-V3
Signup: https://platform.deepseek.com
Minimum credit: $5-10 (much cheaper than OpenAI)
```

**Environment setup**:
```bash
# .env file
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 7.3 Hugging Face (for local Llama)

```
Required for: Llama 3.1 8B Instruct (local inference)
Signup: https://huggingface.co
License: Must accept Meta's Llama license agreement
Token: Required for downloading model weights
```

---

## 8. Experimental Design

### 8.1 Experiments to Run (for the paper)

| Experiment | What it Tests | Comparison |
|------------|---------------|------------|
| **E1**: RAG-AD vs AD-LLM baseline | Does retrieval help? | AD-LLM Table 1 results |
| **E2**: RAG-AD ablation on k | How many exemplars is optimal? | k = 0, 1, 2, 3, 5 |
| **E3**: Self-consistency vs single-pass | Does multi-path help? | m = 1 vs m = 3, 5, 7 |
| **E4**: Aggregation methods | Mean vs median vs weighted | All three compared |
| **E5**: Uncertainty analysis | Are uncertain samples harder? | Accuracy at different uncertainty thresholds |
| **E6**: Explanation quality | Which LLM explains best? | Across 3 LLMs |
| **E7**: RAG + self-consistency combined | Full RAD-LLM pipeline | vs each component alone |
| **E8**: Selective prediction | Can uncertainty improve precision? | Coverage vs accuracy curves |

### 8.2 Evaluation Metrics

**Detection quality** (same as AD-LLM for fair comparison):
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)

**New metrics introduced by RAD-LLM**:
- Explanation Quality Score (EQ): average of faithfulness, specificity, informativeness
- Uncertainty Calibration: Expected Calibration Error (ECE)
- Selective Prediction: AUROC@coverage (e.g., AUROC when only predicting on 80% most confident samples)

### 8.3 Baselines to Compare Against

| Baseline | Source |
|----------|--------|
| 18 traditional AD methods | AD-LLM Table A8 (already computed) |
| AD-LLM zero-shot (Llama 3.1) | Reproduce or cite Table 1 |
| AD-LLM zero-shot (GPT-4o) | Reproduce or cite Table 1 |
| AD-LLM zero-shot (DeepSeek-V3) | Reproduce or cite Table 1 |
| AD-LLM + description augmentation | Reproduce or cite Table 2 |
| RAD-LLM (RAG only) | Our experiment E1 |
| RAD-LLM (self-consistency only) | Our experiment E3 |
| RAD-LLM (full pipeline) | Our experiment E7 |

---

## 9. Project Code Structure

```
RAD-LLM/
│
├── config/
│   ├── datasets.yaml              # Dataset configs (categories, paths, etc.)
│   ├── models.yaml                # LLM configs (API keys, model names)
│   └── experiments.yaml           # Experiment configs (k, m, temperatures)
│
├── data/
│   ├── raw/                       # Original datasets from NLPADBench
│   │   ├── AG_News/
│   │   ├── BBC_News/
│   │   ├── IMDB_Reviews/
│   │   ├── N24_News/
│   │   └── SMS_Spam/
│   ├── processed/                 # Preprocessed data
│   └── embeddings/                # Cached embeddings for RAG
│       └── faiss_indices/         # Pre-built FAISS indices
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              # Load and preprocess datasets
│   │   └── splitter.py            # Train/test/validation splits
│   │
│   ├── retrieval/                 # Component 1: RAG-AD
│   │   ├── __init__.py
│   │   ├── embedder.py            # Text embedding with sentence-transformers
│   │   ├── index_builder.py       # Build FAISS index from training data
│   │   └── retriever.py           # Retrieve top-k similar normal samples
│   │
│   ├── detection/                 # Component 2: Self-Consistency Detection
│   │   ├── __init__.py
│   │   ├── prompts.py             # All prompt templates (base + RAG-enhanced)
│   │   ├── llm_client.py          # Unified LLM interface (GPT-4o, Llama, DeepSeek)
│   │   ├── detector.py            # Single-pass detection (AD-LLM baseline)
│   │   ├── multi_path_detector.py # Multi-path self-consistency detection
│   │   └── aggregator.py          # Score aggregation (mean, median, weighted)
│   │
│   ├── evaluation/                # Component 3: Explanation Evaluation
│   │   ├── __init__.py
│   │   ├── judge.py               # LLM-as-a-Judge for explanation quality
│   │   ├── metrics.py             # AUROC, AUPRC, ECE, selective prediction
│   │   └── uncertainty.py         # Uncertainty calibration analysis
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── rad_llm.py             # Full RAD-LLM pipeline (all components)
│       └── baseline.py            # AD-LLM baseline reproduction
│
├── experiments/
│   ├── run_baseline.py            # Reproduce AD-LLM results
│   ├── run_rag_ablation.py        # E1 + E2: RAG-AD experiments
│   ├── run_consistency.py         # E3 + E4: Self-consistency experiments
│   ├── run_explanation_eval.py    # E6: Explanation quality analysis
│   ├── run_full_pipeline.py       # E7: Full RAD-LLM
│   └── run_selective_prediction.py # E8: Uncertainty-based selective prediction
│
├── results/                       # Experiment outputs
│   ├── tables/                    # LaTeX tables for paper
│   ├── figures/                   # Generated plots for paper
│   └── raw/                       # Raw JSON results
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_rag_analysis.ipynb
│   ├── 03_results_visualization.ipynb
│   └── 04_paper_figures.ipynb
│
├── paper/                         # LaTeX source for the paper
│   ├── main.tex
│   ├── sections/
│   ├── figures/
│   └── tables/
│
├── requirements.txt
├── setup.py
├── README.md
└── .env                           # API keys (git-ignored)
```

---

## 10. Timeline Estimate (College Project)

| Week | Tasks |
|------|-------|
| Week 1 | Set up environment, download datasets, reproduce AD-LLM baseline |
| Week 2 | Implement RAG-AD module (embedding, FAISS index, retrieval) |
| Week 3 | Implement self-consistency scoring module |
| Week 4 | Implement explanation evaluation module |
| Week 5 | Run experiments E1-E4 (RAG ablations, self-consistency) |
| Week 6 | Run experiments E5-E8 (uncertainty, explanation quality, full pipeline) |
| Week 7 | Generate figures, tables, analysis |
| Week 8 | Write paper, review, submit |

---

## 11. Expected Contributions (for the paper)

1. **RAG-AD**: First retrieval-augmented approach for LLM-based zero-shot anomaly detection — grounds LLM reasoning in actual data rather than abstract category labels

2. **Self-Consistency Scoring**: First application of multi-path reasoning aggregation to anomaly scoring — provides both better scores and built-in uncertainty quantification

3. **Explanation Quality Evaluation**: First systematic assessment of LLM-generated anomaly detection explanations using automated metrics — establishes a new evaluation dimension for interpretable AD

4. **Selective Prediction**: First uncertainty-aware selective prediction framework for LLM-based AD — enables practical deployment where uncertain samples are routed to human experts

5. **Comprehensive empirical study**: Ablation studies across 5 datasets, 3 LLMs, multiple configurations demonstrating when and why each component helps

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| API costs exceed budget | Use GPT-4o-mini for most experiments; GPT-4o only for final results |
| Llama 3.1 infinite loops | Implement timeout + retry logic; report error rates as AD-LLM does |
| RAG doesn't improve performance | Still publishable as a negative/nuanced result — analyze WHY |
| Self-consistency too slow | Reduce m to 3; report cost-accuracy tradeoffs |
| Judge LLM biased in evaluation | Use different model for judging than detecting; report inter-LLM agreement |
