# Connector-Aware Pretraining (Phase 2: AGA)

This repository contains the implementation of **Connector-Aware Pretraining**, an optimization strategy designed to improve multi-step logical reasoning in Large Language Models (LLMs) by counteracting "Gradient Starvation" of sparse discourse connectors.

## 🚀 Phase 2 Upgrades: Adaptive Gradient Amplification (AGA)

We have officially migrated from the "Phase 1: XML Tagging" approach to a highly mathematically sound **Phase 2: AGA** methodology.

### The Problem with Phase 1 (XML Tagging)
Initially, we used explicit XML tags (e.g., `<connector type="CAUSAL"> because </connector>`) and a static $1.1\times$ multiplier in the forward pass. This resulted in two critical failures:
1.  **Residual Stiffness:** Multiplying embeddings in the forward pass inflated the residual stream, suppressing the contribution of learned contextual updates.
2.  **OOD Inference Drop:** The model learned to reason *only* when XML tags were present. Removing them during benchmarks (like MMLU) caused a massive Out-of-Distribution shock (-15% accuracy drop).

### The Phase 2 Solution (This Codebase)

1.  **Invisible Masking**: We entirely removed XML tag pollution. The dataset remains pure, raw text. The `ConnectorAwareTrainer` dynamically scans the tokenized sequence (`input_ids`) to identify the token IDs of 150+ known logical connectors on the fly.
2.  **Adaptive Gradient Amplification (AGA)**: Instead of a static $1.1\times$ boost, the trainer computes a dynamic multiplier based on the inverse frequency of the connector in the batch.
    *   $\gamma = 1.0 + \min(0.20, \frac{\alpha}{\text{Frequency}})$
    *   Common connectors ("because") get a small nudge ($\sim 1.05\times$).
    *   Rare connectors ("nevertheless") get a strong nudge (up to $1.20\times$) to prevent gradient starvation.
3.  **Backward Pass Hook**: Instead of multiplying the embedding in the forward pass (which breaks the model's math), we use a `register_hook` on the embedding layer. The AGA multiplier is applied **strictly to the backward gradient**. The model's forward inference remains $100\%$ structurally identical to base Llama 3.2, ensuring flawless zero-shot evaluation on standard benchmarks without requiring special trigger tokens.

## 📁 Repository Structure

*   **`pretrain/model.py`**: Contains the modified `Llama3Model` with the `aga_backward_hook` implementation.
*   **`pretrain/trainer.py`**: Contains the `ConnectorAwareTrainer` which computes the dynamic AGA mask using "Invisible Masking" directly from `input_ids`.
*   **`utils/connector_detector.py`**: The raw text parsing logic (cleared of old XML tag generation).

## 🧠 Comparison to "Thoughts of Words" (ToW)
Unlike ToW, which is an *External/Generative* approach that increases inference latency by forcing the model to generate text "thoughts", AGA is a pure *Optimization/Mechanistic* intervention. We improve reasoning by fundamentally correcting the learning dynamics of the transformer, resulting in a smarter model with zero inference overhead.
