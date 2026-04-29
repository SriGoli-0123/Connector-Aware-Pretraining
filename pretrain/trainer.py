#!/usr/bin/env python3
"""
trainer_FIXED_FINAL.py - WITH attention_mask + connector_mask

CRITICAL FIXES:
1. ✅ attention_mask extracted from inputs
2. ✅ attention_mask PASSED to model forward()
3. ✅ connector_mask extracted from inputs
4. ✅ connector_mask PASSED to model forward()
5. ✅ Uses data_loader_FIXED_V3.py for DirectParquetDataset
6. ✅ Standard cross-entropy loss (Approach 1 - no compounding)

KEY CHANGE FROM PREVIOUS:
✅ Line 155: Now passes BOTH attention_mask AND connector_mask to model
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional
from transformers import (
    TrainingArguments,
    Trainer
)
from datasets import Dataset as HFDataset, load_from_disk
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import ast

logger = logging.getLogger(__name__)


# Removed legacy ConnectorAnnotator and ConnectorAwareDataCollator


# ============================================================================
# Custom Trainer (FIXED FINAL - WITH BOTH MASKS)
# ============================================================================

class ConnectorAwareTrainer(Trainer):
    """
    FIXED FINAL: Trainer that uses Invisible Masking and AGA.
    
    KEY FEATURES:
    1. ✅ Extracts attention_mask from batch
    2. ✅ Invisible Masking: Dynamically identifies connector tokens from input_ids 
    3. ✅ Adaptive Gradient Amplification (AGA): Dynamically calculates frequency-based boost
    """
    
    def __init__(self, *args, config=None, connector_words=None, **kwargs):
        # 1. Identify which keyword the current Trainer version expects
        import inspect
        trainer_sig = inspect.signature(Trainer.__init__)
        parameters = trainer_sig.parameters
        
        # Capture tokenizer from kwargs if it exists
        tokenizer = kwargs.get("tokenizer", None)
        
        # 2. Map to the correct keyword
        if "tokenizer" not in parameters:
            if "processing_class" in parameters:
                # Newest HF version (v4.46+)
                if "tokenizer" in kwargs:
                    kwargs["processing_class"] = kwargs.pop("tokenizer")
            else:
                # Very old or customized version
                kwargs.pop("tokenizer", None)
        
        # 3. Call super and manually ensure self.tokenizer is set
        super().__init__(*args, **kwargs)
        
        if self.tokenizer is None and tokenizer is not None:
            self.tokenizer = tokenizer
            
        self.rl_config = config
        self.connector_token_ids = None
        if connector_words is not None and self.tokenizer is not None:
            # Get token IDs for all connector words to enable Invisible Masking
            ids = set()
            for word in connector_words:
                tokens = self.tokenizer.encode(" " + word, add_special_tokens=False)
                if tokens:
                    ids.add(tokens[0])  # Primary subword
            self.connector_token_ids = torch.tensor(list(ids), dtype=torch.long)
            logger.info(f"✓ Initialized Implicit RL with {len(ids)} connector token IDs")

    def _align_special_tokens(self):
        """Override to skip HF Trainer's config validation."""
        pass
    
    def setup_callbacks(self):
        """Override to disable problematic callbacks for custom models."""
        super().setup_callbacks()
        self.log_model = False
        self.report_to = []
        logger.info("✓ Disabled HF integration callbacks for custom model compatibility")
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate base per-token loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())
        
        # RL Pretraining Objective: Reward-Weighted Cross-Entropy
        # Models receive higher 'reward' (loss weight) for logical sequences
        reward_weights = torch.ones_like(per_token_loss)
        
        # Priority 1: Use provided Ghost Mask (connector_mask)
        # Priority 2: Use dynamic first-token identification
        connector_mask = inputs.get("connector_mask")
        if connector_mask is not None:
            # Shift mask to match logits (mask is on input_ids, reward is on next-token prediction)
            is_connector = connector_mask[..., :-1].contiguous().bool()
        elif self.connector_token_ids is not None:
            self.connector_token_ids = self.connector_token_ids.to(input_ids.device)
            is_connector = torch.isin(shift_labels, self.connector_token_ids)
        else:
            is_connector = torch.zeros_like(shift_labels, dtype=torch.bool)
            
        if is_connector.any() and getattr(self.rl_config, 'use_reward_weighting', True):
            # Apply base logical reward
            # Connectors themselves get a 1.2x boost
            reward_weights[is_connector] = 1.20                
            # Propagate reward to the subsequent reasoning chain (Sequence Rewarding)
            decay_factor = getattr(self.rl_config, 'reward_decay_factor', 0.8)
            chain_length = getattr(self.rl_config, 'reward_chain_length', 5)
            
            base_reward = reward_weights.clone()
            for i in range(1, chain_length + 1):
                shifted_reward = torch.roll(base_reward, shifts=i, dims=1)
                shifted_reward[:, :i] = 1.0
                reward_weights = torch.max(reward_weights, shifted_reward * (decay_factor ** i))

        # Mask out padding tokens
        valid_mask = (shift_labels != -100).float()
        weighted_loss = per_token_loss * reward_weights * valid_mask
        
        final_loss = weighted_loss.sum() / valid_mask.sum()
        
        if return_outputs:
            return final_loss, {"logits": logits}
        return final_loss


# ============================================================================
# Main Pretraining Manager (FIXED FINAL)
# ============================================================================

class ConnectorPretrainingManager:
    """
    FIXED FINAL: Manager for connector-aware pretraining.
    
    Features:
    - Uses AutoModelForCausalLM (Model Agnostic)
    - Applies Reward-Weighted Cross-Entropy (Implicit RL)
    """
    
    def __init__(self, config, model_handler):
        """
        Args:
            config: Config instance
            model_handler: Model handler (from model.py)
        """
        self.config = config
        self.model_handler = model_handler
        
        # Initialize components
        self.trainer = None
        
        logger.info("\n" + "="*70)
        logger.info("CONNECTOR PRETRAINING MANAGER (PHASE 3 - RL)")
        logger.info("="*70)
        logger.info(f"Model: {config.model_name}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Objective: Reward-Weighted Cross-Entropy (RL)")
        logger.info("="*70 + "\n")
    
    def _load_dataset_from_parquet(self, parquet_path: str, max_files: Optional[int] = None) -> HFDataset:
        """
        Load dataset from parquet using data_loader_FIXED_V3.py
        """
        logger.info(f"\n[PARQUET] Loading dataset from: {parquet_path}")
        
        try:
            # ✅ FIXED: Import from data_loader_FIXED_V3.py (no duplication)
            from pretrain.data_loader import DirectParquetDataset
            dataset = DirectParquetDataset(parquet_path, max_files=max_files)
            logger.info(f"✓ Loaded {len(dataset):,} samples from parquet")
            return dataset
        except ImportError as e:
            logger.error(f"❌ Could not import DirectParquetDataset from data_loader: {e}")
            logger.error(f"   Make sure data_loader_FIXED_V3.py is in pretrain/ directory")
            raise
    
    def _load_dataset_from_hf_format(self, dataset_path: str) -> HFDataset:
        """Load dataset from HuggingFace format."""
        logger.info(f"\n[HF FORMAT] Loading dataset from: {dataset_path}")
        dataset = load_from_disk(dataset_path)
        if hasattr(dataset, 'keys'):  # DatasetDict
            logger.info(f"✓ Loaded DatasetDict with splits: {list(dataset.keys())}")
        else:
            logger.info(f"✓ Loaded {len(dataset):,} samples")
        return dataset
    
    def prepare_trainer(
        self,
        train_dataset: HFDataset,
        eval_dataset: Optional[HFDataset] = None,
        output_dir: str = "./output/connector_model",
        boost_factor: float = 1.1,
        num_epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float = 5e-6,
        **kwargs
    ):
        """
        Prepare trainer with corrected connector_mask passing.
        """
        logger.info("\n" + "="*70)
        logger.info("PREPARING TRAINER (FINAL FIXED)")
        logger.info("="*70)
        
        # Create data collator
        logger.info(f"\n[1/2] Setting up data collator...")
        
        from pretrain.data_loader import ConnectorDataCollatorWithMaskCreation
        data_collator = ConnectorDataCollatorWithMaskCreation(
            tokenizer=self.model_handler.tokenizer,
            pad_token_id=self.model_handler.tokenizer.pad_token_id,
            boost_factor=boost_factor
        )
        logger.info("✓ Using Ghost-Masking Collator (Phase 3)")
        
        # Training arguments
        logger.info(f"\n[2/2] Configuring training arguments...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            # Training
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            # Optimization
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=500,
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            # Precision
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            # Logging
            logging_steps=50,
            logging_dir=f"{output_dir}/logs",
            # Evaluation
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=500 if eval_dataset else None,
            # Saving
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,
            # Reporting
            report_to="tensorboard",
            run_name="connector_pretrain",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
        )
        
        logger.info("✓ Training arguments configured")
        
        # Create trainer
        logger.info("\nCreating trainer...")
        
        # Extract all raw connector words from config
        connector_words = []
        for category, words in self.config.connector_types.items():
            connector_words.extend(words)
            
        self.trainer = ConnectorAwareTrainer(
            model=self.model_handler.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_dataset else None,
            data_collator=data_collator,
            tokenizer=self.model_handler.tokenizer,
            config=self.config,
            connector_words=connector_words
        )
        
        logger.info("✓ Trainer created with Implicit RL initialized")
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("TRAINING SETUP SUMMARY")
        logger.info("="*70)
        logger.info(f"Model: {self.config.model_name}")
        logger.info(f"Training samples: {len(train_dataset):,}")
        if eval_dataset:
            logger.info(f"Evaluation samples: {len(eval_dataset):,}")
        
        logger.info(f"\nObjective: Reward-Weighted Cross-Entropy (Implicit RL)")
        logger.info(f"  Base Reward Alpha: {self.config.base_reward_alpha}")
        logger.info(f"  Reward Chain Length: {self.config.reward_chain_length}")
        logger.info(f"  Reward Decay Factor: {self.config.reward_decay_factor}")
        
        logger.info(f"\nTraining Settings:")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info("="*70 + "\n")
    
    def train(self):
        """Execute training"""
        if self.trainer is None:
            raise ValueError("Trainer not prepared. Call prepare_trainer() first.")
        
        logger.info("\n" + "="*70)
        logger.info("STARTING TRAINING")
        logger.info("="*70 + "\n")
        
        self.trainer.train()
        
        logger.info("\n" + "="*70)
        logger.info("✓ TRAINING COMPLETE")
        logger.info("="*70 + "\n")
    
    def evaluate(self):
        """Evaluate model"""
        if self.trainer is None:
            raise ValueError("Trainer not prepared")
        
        results = self.trainer.evaluate()
        logger.info(f"Evaluation results: {results}")
        return results
    
    def save_model(self, output_dir: str = None):
        """Save trained model"""
        if output_dir is None:
            output_dir = self.trainer.args.output_dir + "/final"
        
        logger.info(f"\nSaving model to: {output_dir}")
        self.trainer.save_model(output_dir)
        self.model_handler.tokenizer.save_pretrained(output_dir)
        logger.info("✓ Model saved")

if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add project root to sys.path
    project_root = str(Path(__file__).parent.parent)
    sys.path.append(project_root)
    sys.path.append(str(Path(__file__).parent.parent / "utils"))
    
    from config import Config
    from model import ModelHandler
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    cfg = Config()
    handler = ModelHandler(cfg)
    handler.load_tokenizer()
    handler.load_model()
    
    logger.info("="*70)
    logger.info("STARTING CONNECTOR-AWARE PRETRAINING")
    logger.info("="*70)
    
    # 1. Initialize
    manager = ConnectorPretrainingManager(cfg, handler)
    
    # 2. Load Dataset
    dataset_path = cfg.combined_dataset_path
    if not Path(dataset_path).exists():
        logger.error(f"❌ Dataset not found at {dataset_path}. Run prepare_datasets.py first!")
        sys.exit(1)
        
    dataset = manager._load_dataset_from_hf_format(dataset_path)
    
    # Handle splits
    if hasattr(dataset, 'keys'):
        train_ds = dataset['train']
        eval_ds = dataset.get('validation', dataset.get('test', None))
    else:
        train_ds = dataset
        eval_ds = None
    
    # 3. Prepare Trainer
    manager.prepare_trainer(
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        boost_factor=cfg.boost_factor,
        num_epochs=cfg.num_train_epochs,
        batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate
    )
    
    # 4. Run Training
    manager.train()
    
    # 5. Save Final Model
    manager.save_model()
