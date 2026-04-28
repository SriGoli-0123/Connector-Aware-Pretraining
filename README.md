# Connector-Aware Pretraining (Phase 3: Implicit RL)

This repository implements **Model-Agnostic Implicit Reinforcement Learning** for improving multi-step logical reasoning in Large Language Models (LLMs). It solves the "Gradient Starvation" of logical connectors without resorting to fragile structural hacks.

## 🚀 Phase 3 Upgrades: Reward-Weighted Cross-Entropy

We have successfully migrated past architectural hooks and XML tagging into a pure, model-agnostic **Reinforcement Learning Pretraining Objective (RLP)**.

### The Problem with Phase 1 & 2
*   **Phase 1 (XML Tagging)**: Failed because injecting XML tags caused an Out-of-Distribution (OOD) collapse during standard inference benchmarks like MMLU.
*   **Phase 2 (Backward Hooks)**: Worked well, but required custom PyTorch architectures (a heavily modified `Llama3Model`). This meant the logic could not be easily ported to test Mistral, Qwen, or other models.

### The Phase 3 Solution (This Codebase)

1.  **HuggingFace Native (Model Agnostic)**: `pretrain/model.py` now directly wraps `AutoModelForCausalLM`. You can plug in *any* causal language model architecture and our optimization will work out of the box.
2.  **Implicit RL via Sequence Rewarding**: Instead of hacking the embedding layer, the `ConnectorAwareTrainer` dynamically calculates a **Reward-Weighted Cross-Entropy Loss**.
    *   **The Reward**: If the model predicts a rare logical connector ("therefore", "consequently"), the loss penalty for the sequence that follows is amplified by a high factor (e.g., $1.20\times$).
    *   **The Reasoning Span**: The reward is decayed over a hyperparameter `chain_length=5` tokens. This ensures the model isn't just learning to predict the word "therefore", but is structurally optimizing for the *actual logical conclusion* that follows it.
3.  **Clean Datasets**: Added `prepare_datasets.py` pipelines for **ProofWriter** and **EntailmentBank**—curated, high-signal logical datasets that bypass the parsing noise found in ArXiv PDFs.

## 📁 Repository Structure

*   **`pretrain/model.py`**: Generalized `AutoModelForCausalLM` wrapper.
*   **`pretrain/trainer.py`**: The core logic. Contains the `ConnectorAwareTrainer` that calculates the Sequence Reward and applies it to the Cross-Entropy loss.
*   **`utils/config.py`**: Hyperparameters for the RL logic (`reward_chain_length`, `reward_decay_factor`).

## 🧠 Why "5 Tokens"? (The Reasoning Span)
By propagating the reward forward over $5$ tokens with a decay factor (e.g., $0.8^n$), we encompass the grammatical "consequent" of a logical operator. 
If we boost only $1$ token, the model overfits on grammar ("the"). If we boost $20$, we dilute the signal. $5$ acts as the empirical "Goldilocks window" for capturing a logical deduction.
