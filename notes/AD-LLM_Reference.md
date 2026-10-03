# AD-LLM: Zero-Shot Anomaly Detection — Complete Implementation Analysis

## 1. Overview

AD-LLM (ACL Findings 2025) benchmarks how LLMs can perform anomaly detection in NLP without any task-specific training. The zero-shot detection task (Task 1) is the core contribution — it uses pre-trained LLMs as direct anomaly detectors through carefully engineered prompts.

The fundamental idea: instead of training a model on labeled data, you describe what "normal" looks like to an LLM and ask it to score how anomalous each test sample is.

---

## 2. Theoretical Foundation

### 2.1 Problem Formulation

Given a test set:

```
D_test = {x_1, x_2, ..., x_n}
```

where each sample x_i belongs to either a **normal category** or an **anomaly category**, the objective is to identify anomalous samples using a pre-trained LLM `f_LLM` in a **zero-shot setting** — meaning no task-specific training data is used.

### 2.2 Two Evaluation Settings

The paper evaluates under two levels of prior knowledge:

**Setting 1: Normal Only**
- Input: Only the normal category name(s) `C_normal`
- Anomaly definition: "anything that does NOT belong to any listed normal category"
- Real-world analogy: You know what normal behavior looks like, but anomalies are unknown/emerging

**Setting 2: Normal + Anomaly**
- Input: Both normal category names `C_normal` AND anomaly category name `C_anomaly`
- Anomaly definition: "anything that belongs to the anomaly category rather than any normal category"
- Real-world analogy: You have some information about what anomalies look like

### 2.3 The Detection Equation

The core equation from the paper (Eq. 1):

```
P = T(x_i, C_normal, C_anomaly*)
(r, s) = f_LLM(P)
```

Where:
- `T(.)` is the prompt template constructor
- `x_i` is the test sample
- `C_normal` are the normal category names
- `C_anomaly*` is the anomaly category (only in Setting 2)
- `P` is the constructed prompt
- `f_LLM` is the LLM (Llama 3.1, GPT-4o, or DeepSeek-V3)
- `r` is the verbal explanation (generated FIRST)
- `s` is the anomaly score between 0 and 1

**Critical design choice**: The explanation `r` is generated BEFORE the anomaly score `s`. This acts as an implicit Chain-of-Thought — the LLM "thinks" through its reasoning before committing to a score. The paper found that Llama 3.1 crashes (always outputs s=0) if you reverse this order.

### 2.4 Why Zero-Shot Works for AD

The theoretical justification rests on three pillars:

1. **Pre-trained world knowledge**: LLMs trained on internet-scale corpora have implicit category knowledge (e.g., they "know" what sports articles vs. tech articles look like)

2. **Semantic reasoning**: LLMs can compare a text sample's content, tone, and topic against category descriptions using natural language understanding

3. **Calibrated verbalization**: By requesting a score between 0-1 with explicit scoring guidelines, the LLM produces pseudo-probabilities that can be used with standard AD metrics (AUROC, AUPRC)

---

## 3. The Prompt Templates (Exact Implementation)

### 3.1 Setting 1: "Normal Only" Prompt (Table A9)

This is the complete prompt template used when only normal categories are known:

```
You are an intelligent and professional assistant that detects anomalies in text data.

## Task:
- Following the rules below, determine whether the given text sample is an anomaly.
  Provide a brief explanation of your reasoning and assign an anomaly confidence
  score between 0 and 1.

## Categories:
- **{normal_category_1}**
- **{normal_category_2}**
- ...

## Rules:
1. **Anomaly Definition**:
   - A text sample is considered an **anomaly** if it does **not** belong to
     **any of the categories** listed above.

2. **Scoring**:
   - Assign an anomaly confidence score between 0 and 1.
   - Use higher scores when you are highly confident in your decision.
   - Use lower scores when you are uncertain or think the text sample is
     **not** an anomaly.

3. **Step-by-step Reasoning** (Chain of Thought):
   - **Step 1**. Read the entire text sample carefully and understand it thoroughly.
   - **Step 2**. Analyze the text sample by comparing its content to each category
     listed in the "Categories" section above, considering factors such as main
     topics, meanings, background, sentiments, etc.
   - **Step 3**. Determine which category the text sample **most closely aligns with**.
     - If it aligns with any category, it is **not** an anomaly.
     - If it does **not** align with any category, it is an anomaly.
   - **Step 4**. Assign an anomaly confidence score based on how confident you are
     that the text sample is an anomaly.

4. **Additional Notes**:
   - A text sample may relate to multiple categories, but it should be classified
     into the **most relevant** one in this task.
   - If you are uncertain whether the text sample **significantly aligns** with
     **any of the anomaly category(ies)**, assume that it does **not**, which
     means it is **not** an anomaly.

5. **Response Format**:
   - Provide responses in a strict **JSON** format with the keys "reason" and
     "anomaly_score."
   - "reason": Your brief explanation of the reasoning in one to three sentences
     logically.
   - "anomaly_score": Your anomaly confidence score between 0 and 1.
   - Ensure the JSON output is correctly formatted, including correct placement
     of commas between key-value pairs.
   - Add a backslash (\) before any double quotation marks (") within the values
     of JSON output for proper parsing.

Text sample:
"{text}"

Response in JSON format:
```

### 3.2 Setting 2: "Normal + Anomaly" Prompt (Table A10)

The key difference (marked in red in the paper) is in the anomaly definition:

```
## Categories:
### Normal Category(ies):
- **{normal_category_1}**
- **{normal_category_2}**
- ...

### Anomaly Category(ies):
- {anomaly_category}

## Rules:
1. **Anomaly Definition**:
   - A text sample is considered an **anomaly** if it belongs to the
     **anomaly category(ies)** rather than **any of the normal category(ies)**
     listed above.
```

And in Step 3:
```
   - **Step 3**. Determine which category the text sample **most closely aligns with**.
     - If it **most closely aligns with** **any of the anomaly category(ies)**,
       it is an **anomaly**.
     - If it **most closely aligns with** **any of the normal category(ies)**
       instead, it is **not** an anomaly.
```

Everything else remains the same.

### 3.3 Prompt Engineering Techniques Used

The prompt design incorporates four key techniques:

1. **Task Information** (Cao et al., 2023): Clear definition of the detection scenario, anomaly definition, and rules to reduce hallucinations.

2. **Explicit Chain-of-Thought**: The 4-step reasoning process (Read -> Analyze -> Determine -> Score) is baked directly into the prompt.

3. **Implicit CoT via explanation-first ordering**: The JSON response requires `"reason"` before `"anomaly_score"`, forcing the LLM to reason before scoring.

4. **Structured output (JSON)**: Strict JSON format with escaping rules ensures parseable, consistent outputs.

---

## 4. Code Implementation (Reconstructed from Paper + Repository Structure)

### 4.1 Project Structure

```
AD-LLM/
├── data/                         # 5 NLP datasets
│   ├── AG_News/
│   ├── BBC_News/
│   ├── IMDB_Reviews/
│   ├── N24_News/
│   └── SMS_Spam/
├── task1_detection/              # Zero-shot detection code
│   ├── zero_shot_detection.py    # Main detection script
│   ├── prompts/                  # Prompt templates
│   └── utils/                    # Helper functions
├── task2_augmentation/           # Data augmentation code
├── task3_model_selection/        # Model selection code
└── requirements.txt
```

### 4.2 Core Zero-Shot Detection Pipeline

Here is the reconstructed implementation based on the paper's description:

```python
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================
# STEP 1: Dataset Loading
# ============================================================
# Each dataset has:
#   - Training set: ONLY normal samples
#   - Test set: Mix of normal + anomaly samples with labels
#   - Category names (normal and anomaly)

def load_dataset(dataset_name):
    """
    Load one of the 5 AD datasets.
    Returns test samples, labels (0=normal, 1=anomaly),
    normal category names, and anomaly category name.
    """
    # Dataset configs from Table A2 in the paper:
    dataset_configs = {
        "AG_News": {
            "normal_categories": ["Sports", "Business", "Sci/Tech"],
            "anomaly_category": "World",
            "test_size": 32109,
            "anomaly_ratio": 0.1177
        },
        "BBC_News": {
            "normal_categories": ["Business", "Politics", "Sport", "Tech"],
            "anomaly_category": "Entertainment",
            "test_size": 579,
            "anomaly_ratio": 0.1071
        },
        "IMDB_Reviews": {
            "normal_categories": ["Positive"],
            "anomaly_category": "Negative",
            "test_size": 8952,
            "anomaly_ratio": 0.1661
        },
        "N24_News": {
            "normal_categories": [
                "Television", "Your Money", "Automobiles", "Science",
                "Economy", "Dance", "Travel", "Technology", "Sports",
                "Movies", "Music", "Real Estate", "Books", "Education",
                "Art & Design", "Theater", "Media", "Style",
                "Global Business", "Well", "Health", "Fashion & Style",
                "Opinion"
            ],
            "anomaly_category": "Food",
            "test_size": 19227,
            "anomaly_ratio": 0.0951
        },
        "SMS_Spam": {
            "normal_categories": ["Non-spam (Ham)"],
            "anomaly_category": "Spam",
            "test_size": 1510,
            "anomaly_ratio": 0.1020
        }
    }

    config = dataset_configs[dataset_name]
    # Load actual text data from files...
    # test_texts = [...]
    # test_labels = [...]  # 0 for normal, 1 for anomaly
    return test_texts, test_labels, config


# ============================================================
# STEP 2: Prompt Construction
# ============================================================

def build_prompt_normal_only(text, normal_categories):
    """
    Build the 'Normal Only' prompt (Setting 1).
    The LLM only knows what normal categories exist.
    Anomaly = anything that doesn't fit.
    """
    categories_str = "\n".join(
        [f"- **{cat}**" for cat in normal_categories]
    )

    prompt = f"""You are an intelligent and professional assistant that detects anomalies in text data.

## Task:
- Following the rules below, determine whether the given text sample is an anomaly. Provide a brief explanation of your reasoning and assign an anomaly confidence score between 0 and 1.

## Categories:
{categories_str}

## Rules:
1. **Anomaly Definition**:
   - A text sample is considered an **anomaly** if it does **not** belong to **any of the categories** listed above.

2. **Scoring**:
   - Assign an anomaly confidence score between 0 and 1.
   - Use higher scores when you are highly confident in your decision.
   - Use lower scores when you are uncertain or think the text sample is **not** an anomaly.

3. **Step-by-step Reasoning** (Chain of Thought):
   - **Step 1**. Read the entire text sample carefully and understand it thoroughly.
   - **Step 2**. Analyze the text sample by comparing its content to each category listed in the "Categories" section above, considering factors such as main topics, meanings, background, sentiments, etc.
   - **Step 3**. Determine which category the text sample **most closely aligns with**.
     - If it aligns with any category, it is **not** an anomaly.
     - If it does **not** align with any category, it is an anomaly.
   - **Step 4**. Assign an anomaly confidence score based on how confident you are that the text sample is an anomaly.

4. **Additional Notes**:
   - A text sample may relate to multiple categories, but it should be classified into the **most relevant** one in this task.
   - If you are uncertain whether the text sample **significantly aligns** with **any of the anomaly category(ies)**, assume that it does **not**, which means it is **not** an anomaly.

5. **Response Format**:
   - Provide responses in a strict **JSON** format with the keys "reason" and "anomaly_score."
   - "reason": Your brief explanation of the reasoning in one to three sentences logically.
   - "anomaly_score": Your anomaly confidence score between 0 and 1.

Text sample:
"{text}"

Response in JSON format:"""
    return prompt


def build_prompt_normal_anomaly(text, normal_categories, anomaly_category):
    """
    Build the 'Normal + Anomaly' prompt (Setting 2).
    The LLM knows both normal and anomaly categories.
    """
    normal_str = "\n".join(
        [f"- **{cat}**" for cat in normal_categories]
    )

    prompt = f"""You are an intelligent and professional assistant that detects anomalies in text data.

## Task:
- Following the rules below, determine whether the given text sample is an anomaly. Provide a brief explanation of your reasoning and assign an anomaly confidence score between 0 and 1.

## Categories:
### Normal Category(ies):
{normal_str}
### Anomaly Category(ies):
- {anomaly_category}

## Rules:
1. **Anomaly Definition**:
   - A text sample is considered an **anomaly** if it belongs to the **anomaly category(ies)** rather than **any of the normal category(ies)** listed above.

2. **Scoring**:
   - Assign an anomaly confidence score between 0 and 1.
   - Use higher scores when you are highly confident in your decision.
   - Use lower scores when you are uncertain or think the text sample is **not** an anomaly.

3. **Step-by-step Reasoning** (Chain of Thought):
   - **Step 1**. Read the entire text sample carefully and understand it thoroughly.
   - **Step 2**. Analyze the text sample by comparing its content to each category listed in the "Categories" section above, considering factors such as main topics, meanings, background, sentiments, etc.
   - **Step 3**. Determine which category the text sample **most closely aligns with**.
     - If it **most closely aligns with** **any of the anomaly category(ies)**, it is an **anomaly**.
     - If it **most closely aligns with** **any of the normal category(ies)** instead, it is **not** an anomaly.
   - **Step 4**. Assign an anomaly confidence score based on how confident you are that the text sample is an anomaly.

4. **Additional Notes**:
   - A text sample may relate to multiple categories, but it should be classified into the **most relevant** one in this task.
   - If you are uncertain whether the text sample **significantly aligns** with **any of the anomaly category(ies)**, assume that it does **not**, which means it is **not** an anomaly.

5. **Response Format**:
   - Provide responses in a strict **JSON** format with the keys "reason" and "anomaly_score."
   - "reason": Your brief explanation of the reasoning in one to three sentences logically.
   - "anomaly_score": Your anomaly confidence score between 0 and 1.

Text sample:
"{text}"

Response in JSON format:"""
    return prompt


# ============================================================
# STEP 3: LLM Inference
# ============================================================

# --- Option A: GPT-4o via OpenAI API ---
from openai import OpenAI

def detect_with_gpt4o(prompt):
    """
    Send prompt to GPT-4o and parse the JSON response.
    Temperature = 0 for deterministic output.
    """
    client = OpenAI(api_key="YOUR_API_KEY")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,      # Deterministic output
        seed=42             # For reproducibility
    )

    response_text = response.choices[0].message.content.strip()

    # Parse JSON response
    try:
        result = json.loads(response_text)
        reason = result["reason"]
        anomaly_score = float(result["anomaly_score"])
    except (json.JSONDecodeError, KeyError):
        # Handle parsing errors — mark as error
        reason = "PARSE_ERROR"
        anomaly_score = None

    return reason, anomaly_score


# --- Option B: Llama 3.1 8B Instruct (Local GPU) ---
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_llama():
    """
    Load Llama 3.1 8B Instruct on a single NVIDIA RTX 6000 Ada GPU (48GB).
    """
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def detect_with_llama(prompt, tokenizer, model):
    """
    Run inference with Llama 3.1.
    CRITICAL: Llama requires "reason" BEFORE "anomaly_score" in JSON output.
    If reversed, Llama crashes and always outputs score = 0.
    """
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            temperature=0.0,   # Greedy decoding
            do_sample=False
        )

    response_text = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    # Parse JSON — watch for infinite loop errors with Llama
    try:
        result = json.loads(response_text)
        reason = result["reason"]
        anomaly_score = float(result["anomaly_score"])
    except (json.JSONDecodeError, KeyError):
        reason = "PARSE_ERROR"
        anomaly_score = None

    return reason, anomaly_score


# --- Option C: DeepSeek-V3 via API ---
def detect_with_deepseek(prompt):
    """
    DeepSeek-V3 accessed through official API.
    Known issues: sometimes fails to return valid JSON,
    occasionally returns incorrect formatting.
    """
    client = OpenAI(
        api_key="YOUR_DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",   # DeepSeek-V3
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        seed=42
    )

    response_text = response.choices[0].message.content.strip()

    try:
        result = json.loads(response_text)
        reason = result["reason"]
        anomaly_score = float(result["anomaly_score"])
    except (json.JSONDecodeError, KeyError):
        reason = "PARSE_ERROR"
        anomaly_score = None

    return reason, anomaly_score


# ============================================================
# STEP 4: Run Detection on Full Test Set
# ============================================================

def run_zero_shot_detection(dataset_name, setting="normal_only", llm="gpt4o"):
    """
    Complete zero-shot AD pipeline.

    Args:
        dataset_name: One of "AG_News", "BBC_News", "IMDB_Reviews",
                     "N24_News", "SMS_Spam"
        setting: "normal_only" or "normal_anomaly"
        llm: "gpt4o", "llama", or "deepseek"
    """
    # Load data
    test_texts, test_labels, config = load_dataset(dataset_name)
    normal_cats = config["normal_categories"]
    anomaly_cat = config["anomaly_category"]

    # Load model if using Llama
    if llm == "llama":
        tokenizer, model = load_llama()

    scores = []
    reasons = []
    errors = 0

    for i, text in enumerate(test_texts):
        # Build prompt based on setting
        if setting == "normal_only":
            prompt = build_prompt_normal_only(text, normal_cats)
        else:
            prompt = build_prompt_normal_anomaly(text, normal_cats, anomaly_cat)

        # Run inference
        if llm == "gpt4o":
            reason, score = detect_with_gpt4o(prompt)
        elif llm == "llama":
            reason, score = detect_with_llama(prompt, tokenizer, model)
        elif llm == "deepseek":
            reason, score = detect_with_deepseek(prompt)

        # Handle errors
        if score is None:
            errors += 1
            continue

        scores.append(score)
        reasons.append(reason)

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(test_texts)} samples, {errors} errors")

    # Filter out error samples from labels too
    valid_labels = [
        label for j, label in enumerate(test_labels)
        if j < len(scores) + errors  # simplified; real impl tracks indices
    ]

    return scores, valid_labels, reasons, errors


# ============================================================
# STEP 5: Evaluation
# ============================================================

def evaluate(scores, labels):
    """
    Compute AUROC and AUPRC — the two metrics used in AD-LLM.
    Both are threshold-free and work directly with continuous scores.
    """
    scores = np.array(scores)
    labels = np.array(labels)

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    return auroc, auprc


# ============================================================
# STEP 6: Main Execution
# ============================================================

if __name__ == "__main__":
    datasets = ["AG_News", "BBC_News", "IMDB_Reviews", "N24_News", "SMS_Spam"]
    settings = ["normal_only", "normal_anomaly"]
    llms = ["gpt4o"]  # or ["llama", "gpt4o", "deepseek"]

    for dataset in datasets:
        for setting in settings:
            for llm in llms:
                print(f"\n{'='*60}")
                print(f"Dataset: {dataset} | Setting: {setting} | LLM: {llm}")
                print(f"{'='*60}")

                scores, labels, reasons, errors = run_zero_shot_detection(
                    dataset, setting, llm
                )

                auroc, auprc = evaluate(scores, labels)

                print(f"AUROC: {auroc:.4f}")
                print(f"AUPRC: {auprc:.4f}")
                print(f"Errors: {errors}")

                # Save results
                results = {
                    "dataset": dataset,
                    "setting": setting,
                    "llm": llm,
                    "auroc": auroc,
                    "auprc": auprc,
                    "errors": errors,
                    "scores": scores,
                    "reasons": reasons
                }

                with open(f"results_{dataset}_{setting}_{llm}.json", "w") as f:
                    json.dump(results, f, indent=2)
```

---

## 5. Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    AD-LLM ZERO-SHOT PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────┐     ┌──────────────────┐     ┌──────────────────┐
  │ Test     │     │ Prompt Template  │     │ Category Names   │
  │ Sample   │     │ (Table A9/A10)   │     │ C_normal,        │
  │ x_i      │     │                  │     │ C_anomaly*       │
  └────┬─────┘     └────────┬─────────┘     └────────┬─────────┘
       │                    │                         │
       └────────────────────┼─────────────────────────┘
                            │
                            v
                   ┌────────────────┐
                   │  T(x_i, C_n,   │     Prompt Constructor
                   │  C_a*) = P     │     fills template with
                   └───────┬────────┘     sample + categories
                           │
                           v
                   ┌────────────────┐
                   │   f_LLM(P)     │     LLM processes prompt
                   │                │     (temp=0, seed=42)
                   │  Llama 3.1     │
                   │  GPT-4o        │
                   │  DeepSeek-V3   │
                   └───────┬────────┘
                           │
                           v
                   ┌────────────────┐
                   │  JSON Output   │
                   │  {             │
                   │   "reason": r, │     reason FIRST
                   │   "anomaly_    │     (implicit CoT)
                   │    score": s   │     then score
                   │  }             │
                   └───────┬────────┘
                           │
                     ┌─────┴──────┐
                     v            v
              ┌──────────┐  ┌──────────┐
              │ reason r │  │ score s  │
              │ (text    │  │ (float   │
              │ explain) │  │ 0 to 1)  │
              └──────────┘  └────┬─────┘
                                 │
                    ┌────────────┘
                    v
           ┌─────────────────┐
           │ Collect scores  │     For ALL test samples
           │ for all x_i     │
           └────────┬────────┘
                    │
                    v
           ┌─────────────────┐
           │ Compute Metrics │
           │ - AUROC         │     Using sklearn
           │ - AUPRC         │     against ground truth
           └─────────────────┘
```

---

## 6. Concrete Example Walkthrough

### Dataset: AG News
- Normal categories: Sports, Business, Sci/Tech
- Anomaly category: World
- Anomaly ratio in test set: 11.77%

### Example 1: Normal sample (Sports article)

**Input text**: `"Tiger Woods wins the Masters tournament in dramatic fashion, finishing 3 under par on the final day."`

**Prompt** (Normal Only setting): The template is filled with categories = [Sports, Business, Sci/Tech] and the text above.

**LLM Response**:
```json
{
  "reason": "The text discusses Tiger Woods winning a golf tournament (the Masters), which clearly falls under the Sports category. Golf is a well-known sport, and the content describes a competitive sporting event.",
  "anomaly_score": 0.05
}
```

Score = 0.05 (low, meaning NOT anomalous) — correct.

### Example 2: Anomaly sample (World news)

**Input text**: `"The United Nations Security Council voted today to impose new sanctions on North Korea following its latest nuclear test."`

**LLM Response**:
```json
{
  "reason": "This text discusses international politics and diplomacy involving the UN Security Council and North Korea nuclear sanctions. It does not align with Sports, Business, or Sci/Tech categories. It most closely relates to world affairs or international relations, which is not listed.",
  "anomaly_score": 0.92
}
```

Score = 0.92 (high, meaning anomalous) — correct.

---

## 7. Key Results from the Paper

### Performance Table (from Table 1)

| LLM | Setting | AG News AUROC | BBC News AUROC | IMDB AUROC | N24 News AUROC | SMS Spam AUROC |
|-----|---------|--------------|----------------|------------|----------------|----------------|
| Llama 3.1 | Normal Only | 0.8226 | 0.7910 | 0.7373 | 0.6267 | 0.7558 |
| Llama 3.1 | Normal+Anomaly | 0.8754 | 0.8612 | 0.8625 | 0.8784 | 0.9487 |
| GPT-4o | Normal Only | 0.9332 | 0.9574 | 0.9349 | 0.7674 | 0.7940 |
| GPT-4o | Normal+Anomaly | 0.9293 | 0.9919 | 0.9668 | 0.9902 | 0.9862 |
| DeepSeek-V3 | Normal Only | 0.9104 | 0.8206 | 0.8544 | 0.8207 | 0.8797 |
| DeepSeek-V3 | Normal+Anomaly | 0.9273 | 0.9581 | 0.9626 | 0.9514 | 0.9535 |
| Best Baseline | — | 0.9226 | 0.9732 | 0.7366 | 0.8320 | 0.9398 |

### Key observations:
1. GPT-4o and DeepSeek-V3 consistently beat traditional baselines
2. Adding anomaly category info (Setting 2) dramatically improves performance
3. Llama 3.1 is competitive but weaker, especially in the Normal Only setting
4. The biggest gains from Setting 1 → 2 are on N24 News (+0.25 AUROC for Llama)

---

## 8. Known Issues and Error Analysis

### Llama 3.1 Infinite Loops
Llama sometimes enters infinite loops, generating the same sentence repeatedly until hitting the token limit. Example from AG News:

```json
{"reason": "The text sample is about a tour company in Australia, which
relates to travel, making it most closely align with the Sports category
is not the best fit, but it does fit into the Sports category is not the
best fit, but it does fit into the category of travel which is not listed,
but the closest is Sports, but it is more closely related to the category
of travel which is not listed, but the closest is Sports..."}
```

Error counts for Llama 3.1: AG News had 552 errors in Normal Only, 48 in Normal+Anomaly.

### GPT-4o Safety Filters
GPT-4o occasionally blocks responses on politically sensitive content. Error counts are low (0-9 per dataset).

### DeepSeek-V3 JSON Failures
DeepSeek sometimes returns malformed JSON or no output at all. More frequent on some datasets (IMDB: 206 errors in Normal Only).

---

## 9. Evaluation Metrics Explained

### AUROC (Area Under ROC Curve)
- Measures the probability that a randomly chosen anomaly sample has a higher score than a randomly chosen normal sample
- Range: 0.5 (random) to 1.0 (perfect)
- Threshold-independent — evaluates across all possible thresholds

### AUPRC (Area Under Precision-Recall Curve)
- More informative when classes are imbalanced (which they are — anomaly ratios are 9-17%)
- Focuses on how well the model ranks anomalies at the top
- Range: anomaly ratio (random baseline) to 1.0 (perfect)

Both are computed directly from the continuous anomaly scores — no threshold selection needed.

---

## 10. Gaps in the Current Implementation (Opportunities for Your Work)

1. **Single-pass inference**: Each sample is scored exactly once at temperature=0 — no uncertainty estimation
2. **No retrieval**: The LLM receives only category names, never actual data samples
3. **Fixed prompt template**: Same prompt for all datasets regardless of their characteristics
4. **No calibration**: Raw verbal scores are used directly — LLMs tend to be overconfident
5. **No ensemble**: Only one LLM is used at a time
6. **No explanation evaluation**: Explanations are generated but never assessed for quality
7. **No few-shot learning**: Zero-shot only, no fine-tuning or in-context examples

These gaps directly map to the novel features proposed for the RAD-LLM framework.
