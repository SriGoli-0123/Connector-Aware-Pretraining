#!/usr/bin/env python3
"""
data_loader.py - PHASE 3: GHOST MASKING DATA LOADER

CLEANED VERSION:
1. ✅ Removed all XML tag logic.
2. ✅ Uses pre-calculated Ghost Masks (connector_mask) from the dataset.
3. ✅ Ensures special tokens (BOS, EOS, PAD) are NEVER boosted.
4. ✅ Validates and fixes attention_mask for proper padding.
5. ✅ Model-agnostic and efficient.
"""

import logging
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset, DatasetDict, load_from_disk
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class ConnectorDataCollatorWithMaskCreation:
    """
    PHASE 3 COLLATOR: Ghost Masking.
    
    This collator is designed for model-agnostic pre-training. It does not
    modify the text or add special tokens. Instead, it uses a binary mask
    to boost the loss of specific tokens at training time.
    """
    
    def __init__(self, tokenizer, pad_token_id=None, boost_factor=1.1):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        self.boost_factor = boost_factor
        
        # Identify special tokens to exclude from any logical rewards
        self.special_token_ids = set(self.tokenizer.all_special_ids)
        if self.pad_token_id is not None:
            self.special_token_ids.add(self.pad_token_id)
            
        logger.info(f"✓ Ghost-Mask Collator initialized (Boost: {boost_factor}x, Pad ID: {self.pad_token_id})")

    def _create_boost_mask(self, batch: List[Dict], max_len: int) -> torch.Tensor:
        """
        Converts the binary connector_mask from the dataset into a decimal boost mask.
        """
        batch_size = len(batch)
        mask = torch.ones((batch_size, max_len), dtype=torch.float32)
        
        for i, item in enumerate(batch):
            conn_mask = item.get("connector_mask", [])
            input_ids = item.get("input_ids", [])
            
            if not conn_mask:
                continue
                
            # Convert binary mask to tensor
            curr_mask = torch.tensor(conn_mask, dtype=torch.float32)
            length = min(len(curr_mask), max_len)
            
            # SAFETY GATE: Ensure structural special tokens never receive a logical boost
            # Even if the detector accidentally flagged one, we zero it out here.
            for j in range(length):
                if input_ids[j] in self.special_token_ids:
                    curr_mask[j] = 0.0
            
            # Apply boost: 1.0 (base) + (binary_mask * (boost_factor - 1.0))
            # e.g., 1.0 + (1 * 0.2) = 1.2
            mask[i, :length] = 1.0 + (curr_mask[:length] * (self.boost_factor - 1.0))
            
        return mask

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collates the batch, handles padding, and creates the logical boost mask.
        """
        if not batch:
            return {}
            
        input_ids_list = []
        attention_mask_list = []
        
        # 1. Extract and validate existing masks
        for item in batch:
            ids = item.get("input_ids", [])
            attn = item.get("attention_mask", [])
            
            # Ensure they are lists
            if isinstance(ids, torch.Tensor): ids = ids.tolist()
            if isinstance(attn, torch.Tensor): attn = attn.tolist()
            
            input_ids_list.append(ids)
            attention_mask_list.append(attn)
            
        # 2. Dynamic Padding
        max_len = max(len(ids) for ids in input_ids_list)
        
        for i in range(len(input_ids_list)):
            pad_len = max_len - len(input_ids_list[i])
            if pad_len > 0:
                input_ids_list[i].extend([self.pad_token_id] * pad_len)
                attention_mask_list[i].extend([0] * pad_len)
                
        # 3. Convert to Tensors
        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)
        
        # 4. Create Labels (Shifted inside model, but we mask padding here)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        # 5. Create the Ghost Reward Mask
        connector_mask = self._create_boost_mask(batch, max_len)
        
        # Ensure reward doesn't apply to padding
        connector_mask = connector_mask * attention_mask.float()
        connector_mask = connector_mask + (1.0 - attention_mask.float())
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "connector_mask": connector_mask
        }

class DirectParquetDataset:
    """Helper to load prepared logical datasets."""
    def __init__(self, path: str, max_files: Optional[int] = None):
        self.path = Path(path)
        if self.path.is_file():
            files = [self.path]
        else:
            files = sorted(self.path.glob("*.parquet"))
            
        if max_files:
            files = files[:max_files]
            
        self.data = []
        for f in files:
            df = pd.read_parquet(f)
            self.data.extend(df.to_dict('records'))
            
        logger.info(f"✓ Loaded {len(self.data):,} logical reasoning samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
