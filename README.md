# ⚡ Fine-Tuning & Model Customization — Comprehensive Assignment Submission

![Model Customization](https://img.shields.io/badge/Fine--Tuning-PEFT%20%7C%20LoRA%20%7C%20QLoRA-blue?logo=pytorch)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers%20v5.13-yellow?logo=huggingface)
![PEFT](https://img.shields.io/badge/PEFT-v0.21.2-orange)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python)
![TOPS Technologies](https://img.shields.io/badge/Program-Generative%20AI%20Specialization-purple)

---

## 📌 Executive Summary

This repository contains the comprehensive, production-tested assignment deliverables for **Fine-Tuning & Model Customization** under the Generative AI Specialization. The curriculum evaluates the technical and architectural spectrum of language model adaptation—from In-Context Prompting and Retrieval-Augmented Generation (RAG) to Parameter-Efficient Fine-Tuning (PEFT) using Low-Rank Adaptation (LoRA).

Every session is organized into dedicated directories containing detailed analytical documentation, runnable Python simulation engines, PyTorch/PEFT training scripts, and serialized adapter weights.

---

## 🗂️ Repository Directory Structure

```text
FINE-TUNING & MODEL CUSTOMIZATION Assignment/
│
├── README.md                                          # Master documentation, cheat sheets & benchmark reports
│
├── Session 1 - Fine-Tuning Introduction & Why Fine-Tune/
│   └── Session_1_Fine_Tuning_Introduction.md          # Tasks 1 to 5: Feature analysis, Gmail spam matrix, RAG vs FT
│
├── Session 2 - Prompt vs RAG vs Fine-Tuning/
│   ├── Session_2_Prompt_vs_RAG_vs_FineTuning.md       # Tasks 1 to 5: Comparison matrix, app use-cases, BookMyShow advice
│   ├── ipl_stats.txt                                  # Verified IPL knowledge base corpus for retrieval
│   └── ipl_rag_simulation.py                          # Runnable RAG workflow script with keyword scoring
│
└── Session 3 - LoRA, PEFT & QLoRA/
    ├── Session_3_LoRA_PEFT_QLoRA.md                   # Tasks 1 to 5: Empirical parameter audits, memory benchmarks
    ├── task1_lora_parameters.py                       # Hugging Face + PEFT parameter count evaluation
    ├── task2_lora_finetune.py                         # 3-epoch custom tweet classification training loop
    ├── task4_memory_benchmark.py                      # RAM & AdamW optimizer state footprint profiler
    ├── task5_chatgpt_lora_snippet.py                  # Production-hardened ChatGPT LoRA execution script
    └── lora_adapted_weights/                          # Saved LoRA adapter weights (3.51 MB total footprint)
        ├── adapter_config.json                        # LoRA rank, alpha, and target modules configuration
        ├── adapter_model.safetensors                  # Adapted low-rank projection weights (2.82 MB)
        ├── tokenizer.json                             # Serialized vocabulary tokenizer
        └── tokenizer_config.json                      # Tokenizer configuration metadata
```

---

## 🚀 Session-by-Session Breakdown & Deliverables

### 📘 [Session 1: Fine-Tuning Introduction & Why Fine-Tune](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%201%20-%20Fine-Tuning%20Introduction%20&%20Why%20Fine-Tune/Session_1_Fine_Tuning_Introduction.md)
* **Task 1 (App Feature Enhancements):** Evaluated how fine-tuning dramatically improves Zomato taste profiling (regional food vocabulary like *"tikha"*, *"dum"*, *"jain"*), Instagram multilingual comment moderation (Hinglish/Gujlish code-mixed slang), and Zomato customer support refund automation.
* **Task 2 (Pre-Trained vs. Fine-Tuned Matrix):** In-depth comparison of Gmail spam detection across Detection Accuracy (80% vs 99.8%), Contextual Relevance (handling regional banking templates), and User Experience (zero inbox clutter).
* **Task 3 (Prompting Failure Boundaries):** Explored high-stakes dispute resolution on Flipkart where prompt engineering fails due to user jailbreaking, context degradation, and non-deterministic policy enforcement, requiring fine-tuning for compliance.
* **Task 4 (Architectural Decision for Gujarat Regional Music):** Justified why Retrieval-Augmented Generation (RAG) is superior to fine-tuning for regional music streaming due to fast-moving releases (Navratri Garba, viral reels) and zero-retraining personalization.
* **Task 5 (Gujarati Food Delivery Chatbot):** Crafted a customized Gujarati food prompt and explained how fine-tuning bakes regional food taxonomies directly into weights, cutting token latency by 60%+.

---

### 📗 [Session 2: Prompt vs RAG vs Fine-Tuning](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%202%20-%20Prompt%20vs%20RAG%20vs%20Fine-Tuning/Session_2_Prompt_vs_RAG_vs_FineTuning.md)
* **Task 1 (Prompt Engineering for Social Media):** Tested in-context prompt generating 3 aesthetic, character-constrained Instagram bios for an Ahmedabad engineering student.
* **Task 2 (RAG Simulation Engine):** Implemented [`ipl_rag_simulation.py`](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%202%20-%20Prompt%20vs%20RAG%20vs%20Fine-Tuning/ipl_rag_simulation.py) searching [`ipl_stats.txt`](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%202%20-%20Prompt%20vs%20RAG%20vs%20Fine-Tuning/ipl_stats.txt) and generating fact-grounded answers for all-time runs, highest team totals, and 2024 champions.
* **Task 3 (Real-World App Scenarios):** Defined clear use cases across Zomato (Prompting for festive greetings), Flipkart (RAG for dynamic flash-sale prices and inventory), and Paytm (Fine-Tuning for real-time fraud transaction classification).
* **Task 4 (Comprehensive Comparison Matrix):** Evaluated Prompting vs RAG vs Fine-Tuning across Setup Cost, Time to Deploy, Maintenance Effort, Data Freshness, and Domain Adaptation depth.
* **Task 5 (BookMyShow Movie Recommendation Bot):** Consulted ChatGPT and established a final architectural decision for a **Hybrid RAG + Prompting engine** to eliminate showtime hallucinations.

---

### 📙 [Session 3: LoRA, PEFT & QLoRA](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%203%20-%20LoRA,%20PEFT%20&%20QLoRA/Session_3_LoRA_PEFT_QLoRA.md)
* **Task 1 (Trainable Parameter Reduction Audit):** Executed [`task1_lora_parameters.py`](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%203%20-%20LoRA,%20PEFT%20&%20QLoRA/task1_lora_parameters.py):
  - Base Model Parameters: **66,955,010**
  - LoRA Adapter Parameters: **739,586**
  - Trainable Parameter Ratio: **1.0925%** (**98.91% parameters frozen**)
* **Task 2 (Tweet Sentiment Fine-Tuning Loop):** Executed [`task2_lora_finetune.py`](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%203%20-%20LoRA,%20PEFT%20&%20QLoRA/task2_lora_finetune.py) training over 3 epochs (loss dropped $0.8782 \rightarrow 0.3082$). Saved adapter weights to disk:
  - Full Model Checkpoint: **~268.00 MB**
  - LoRA Adapter Weights (`adapter_model.safetensors`): **2.82 MB (2,892.77 KB)**
  - Total Adapter Directory: **3.51 MB** (**98.7% storage footprint reduction**)
* **Task 3 (PEFT Value Proposition & App Scenarios):** Documented core benefits (compute savings, catastrophic forgetting prevention, multi-tenant serving) with real-world mobile app architectures for **WhatsApp** (on-device local predictive typing) and **Zomato** (multi-task adapter routing).
* **Task 4 (RAM & Optimizer Memory Profiling):** Executed [`task4_memory_benchmark.py`](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%203%20-%20LoRA,%20PEFT%20&%20QLoRA/task4_memory_benchmark.py) demonstrating an **89x reduction in AdamW optimizer memory states** (510.83 MB for full fine-tuning vs. 5.64 MB for LoRA).
* **Task 5 (AI Code Hardening):** Executed [`task5_chatgpt_lora_snippet.py`](file:///C:/Users/DHRUV%20MODI/Documents/TOPS/assignment/FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION/%20FINE-TUNING%20%26%20MODEL%20CUSTOMIZATION%20Assignment/Session%203%20-%20LoRA,%20PEFT%20&%20QLoRA/task5_chatgpt_lora_snippet.py) detailing the 3 essential modifications required to fix PEFT `target_modules` exceptions in modern library versions.

---

## 📊 Summary Comparison: Customization Approaches

| Feature | In-Context Prompting | Retrieval-Augmented Generation (RAG) | Parameter-Efficient Fine-Tuning (PEFT/LoRA) | Full Model Fine-Tuning |
| :--- | :---: | :---: | :---: | :---: |
| **Model Weights Changed?** | No (Frozen) | No (Frozen) | Yes (Low-Rank Adapters Only) | Yes (All Parameters Updated) |
| **Data Freshness** | Static | **Real-Time Dynamic** | Static (Retraining required) | Static (Retraining required) |
| **GPU Compute Needed** | None (Inference only) | Low (Embedding models) | **Low to Moderate (Single GPU)** | High (Multi-GPU clusters) |
| **Storage per Task** | 0 MB | Vector Database Index | **~3 MB to 30 MB** | 500 MB to 100+ GB |
| **Hallucination Risk** | High | **Near Zero (Grounded)** | Low for learned patterns | Low for learned patterns |

---

## 🧪 Verification & Execution Commands

Run any of the Python engines and benchmarks from your terminal:

```powershell
# 1. Run the IPL RAG Simulation Engine
python "Session 2 - Prompt vs RAG vs Fine-Tuning/ipl_rag_simulation.py"

# 2. Run the LoRA Parameter Efficiency Audit (Task 1)
python "Session 3 - LoRA, PEFT & QLoRA/task1_lora_parameters.py"

# 3. Train the LoRA Text Classification Model on Tweets (Task 2)
python "Session 3 - LoRA, PEFT & QLoRA/task2_lora_finetune.py"

# 4. Run the RAM & Optimizer Memory Profiling Benchmark (Task 4)
python "Session 3 - LoRA, PEFT & QLoRA/task4_memory_benchmark.py"

# 5. Execute the Hardened ChatGPT LoRA Snippet (Task 5)
python "Session 3 - LoRA, PEFT & QLoRA/task5_chatgpt_lora_snippet.py"
```

---

## 👨‍💻 Submission & Verification Details
- **Subject:** Fine-Tuning & Model Customization (PEFT / LoRA / QLoRA)
- **Environment:** Python 3.13 | PyTorch 2.13.0+cpu | Transformers 5.13.0 | PEFT 0.21.2
- **Status:** All 15 Tasks Across 3 Sessions Fully Completed, Executed, and Documented
