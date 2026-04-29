#!/usr/bin/env python3
"""
model.py - PHASE 3: Generalized HuggingFace Model Handler

CRITICAL UPDATES:
✅ Removed custom Llama3Model architecture.
✅ Generalized to use AutoModelForCausalLM.
✅ Can now be used with ANY causal language model (Mistral, Qwen, Gemma, Llama).
✅ RL Logic is handled entirely in trainer.py.
"""

import torch
import logging
from pathlib import Path
from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class ModelHandler:
    """Generalized Model Handler for HuggingFace Causal LMs."""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.original_vocab_size = None
        
        logger.info(f"✓ Model handler initialized")
        logger.info(f"  Model: {config.model_name}")
        logger.info(f"  Device: {config.device}")
    
    def load_tokenizer(self):
        """Load tokenizer for the specified model."""
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.original_vocab_size = len(self.tokenizer)
        logger.info(f"✓ Tokenizer loaded - vocab size: {self.original_vocab_size:,}")
        return self.tokenizer
    
    def load_model(self):
        """Load AutoModelForCausalLM."""
        logger.info(f"Loading Model: {self.config.model_name}")
        
        dtype = getattr(torch, self.config.torch_dtype, torch.float32)
        
        # Load natively generalized HF model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else "sdpa"
        )
        
        device = torch.device(self.config.device)
        self.model = self.model.to(device)
        
        logger.info(f"✓ Model loaded on {device}")
        return self.model
    
    def get_model_info(self) -> Dict:
        """Get model statistics."""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "vocab_size": len(self.tokenizer),
            "device": str(next(self.model.parameters()).device),
        }
    
    def print_model_info(self):
        """Print model information."""
        info = self.get_model_info()
        if not info:
            return
            
        logger.info("\n" + "="*70)
        logger.info("MODEL INFORMATION")
        logger.info("="*70)
        logger.info(f"Model: {info['model_name']}")
        logger.info(f"Parameters: {info['total_parameters']:,} ({info['trainable_parameters']:,} trainable)")
        logger.info(f"Vocab Size: {info['vocab_size']:,}")
        logger.info(f"Device: {info['device']}")
        logger.info("="*70 + "\n")
    
    def save_model(self, output_path: str):
        """Save model and tokenizer."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"✓ Model and tokenizer saved")
    
    def load_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Load from checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        if not checkpoint_dir.exists():
            logger.error(f"Checkpoint not found: {checkpoint_dir}")
            return False
            
        logger.info(f"Loading from checkpoint: {checkpoint_dir}")
        
        dtype = getattr(torch, self.config.torch_dtype, torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir, 
            torch_dtype=dtype,
            attn_implementation="flash_attention_2" if self.config.use_flash_attention else "sdpa"
        )
        
        device = torch.device(self.config.device)
        self.model = self.model.to(device)
        logger.info(f"✓ Model loaded")
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        logger.info(f"✓ Tokenizer loaded")
        
        return True


def initialize_model(config) -> ModelHandler:
    """Initialize model from config."""
    logger.info("\n" + "="*70)
    logger.info(f"INITIALIZING {config.model_name.upper()}")
    logger.info("="*70)
    
    model_handler = ModelHandler(config)
    
    logger.info("\n[1/2] Loading tokenizer...")
    model_handler.load_tokenizer()
    
    logger.info("\n[2/2] Loading model...")
    model_handler.load_model()
    
    model_handler.print_model_info()
    
    return model_handler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        from utils.config import Config
        config = Config()
        config.print_summary()
        
        model_handler = initialize_model(config)
        logger.info("✓ Model initialization successful!")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

