"""
Enhanced Data Loader for VV-GPT
Supports plain text (books, novels) and WhatsApp conversation formats
"""

import re
import json
import torch
import tiktoken
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import unicodedata

class DataProcessor:
    """Handles different data formats and preprocessing"""
    
    def __init__(self, vocab_size: int = None):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.encode = None
        self.decode = None
        # Keep these for backwards compatibility with old checkpoints
        self.chars = None
        self.stoi = None
        self.itos = None
        
    def load_and_process_data(self, file_path: str, data_type: str = 'auto') -> Tuple[str, Dict]:
        """
        Load and process data from file
        
        Args:
            file_path: Path to the data file
            data_type: 'plain_text', 'whatsapp', or 'auto' for auto-detection
        
        Returns:
            Tuple of (processed_text, metadata)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Auto-detect format if needed
        if data_type == 'auto':
            data_type = self._detect_format(raw_text)
            print(f"Auto-detected format: {data_type}")
        
        # Process based on format
        if data_type == 'whatsapp':
            processed_text, metadata = self._process_whatsapp(raw_text)
        else:  # plain_text
            processed_text, metadata = self._process_plain_text(raw_text)
        
        # Clean and prepare text
        processed_text = self._clean_text(processed_text)
        
        return processed_text, metadata
    
    def _detect_format(self, text: str) -> str:
        """Auto-detect if text is WhatsApp format or plain text"""
        # WhatsApp exports typically have timestamp patterns
        whatsapp_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}',  # 12/31/23, 10:30
            r'\[\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}:\d{2}\]',  # [12/31/23, 10:30:45]
            r'\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}',  # 31-12-2023 10:30
        ]
        
        for pattern in whatsapp_patterns:
            if re.search(pattern, text[:1000]):  # Check first 1000 chars
                return 'whatsapp'
        
        return 'plain_text'
    
    def _process_whatsapp(self, text: str) -> Tuple[str, Dict]:
        """Process WhatsApp conversation export"""
        lines = text.strip().split('\n')
        conversations = []
        
        # Common WhatsApp timestamp patterns
        timestamp_patterns = [
            r'^(\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.+)$',
            r'^\[(\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}(?::\d{2})?)\]\s*([^:]+):\s*(.+)$',
            r'^(\d{1,2}-\d{1,2}-\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)\s*-\s*([^:]+):\s*(.+)$',
        ]
        
        current_message = ""
        current_sender = ""
        participants = set()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match timestamp patterns
            matched = False
            for pattern in timestamp_patterns:
                match = re.match(pattern, line)
                if match:
                    # Save previous message if exists
                    if current_message and current_sender:
                        conversations.append(f"{current_sender}: {current_message}")
                        participants.add(current_sender)
                    
                    # Start new message
                    timestamp, sender, message = match.groups()
                    current_sender = sender.strip()
                    current_message = message.strip()
                    matched = True
                    break
            
            if not matched and current_message:
                # This is a continuation of the previous message
                current_message += " " + line
        
        # Don't forget the last message
        if current_message and current_sender:
            conversations.append(f"{current_sender}: {current_message}")
            participants.add(current_sender)
        
        # Format as conversation
        formatted_text = "\n".join(conversations)
        
        # Add special conversation tokens
        formatted_text = "<|start_conversation|>\n" + formatted_text + "\n<|end_conversation|>"
        
        metadata = {
            'format': 'whatsapp',
            'participants': list(participants),
            'total_messages': len(conversations),
            'file_size': len(text)
        }
        
        return formatted_text, metadata
    
    def _process_plain_text(self, text: str) -> Tuple[str, Dict]:
        """Process plain text (books, novels, etc.)"""
        # For plain text, we can add some structure markers
        paragraphs = text.split('\n\n')
        
        # Add paragraph markers for better structure learning
        formatted_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para:
                formatted_paragraphs.append(f"<|paragraph|>\n{para}")
        
        formatted_text = "\n\n".join(formatted_paragraphs)
        
        metadata = {
            'format': 'plain_text',
            'paragraphs': len(formatted_paragraphs),
            'words': len(text.split()),
            'file_size': len(text)
        }
        
        return formatted_text, metadata
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove or replace problematic characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r' {2,}', ' ', text)      # Max 1 space between words
        
        # Remove some problematic characters that might cause issues
        text = re.sub(r'[^\w\s\n.,!?;:(){}[\]"\'-<>|]', '', text)
        
        return text.strip()
    
    def build_vocabulary(self, text: str = None) -> None:
        """Initialize tiktoken vocabulary"""
        print("Initializing tiktoken BPE vocabulary (gpt2)...")
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.tokenizer.n_vocab
        print(f"Vocabulary size: {self.vocab_size} tokens")
        
        # Create encode/decode functions
        self.encode = lambda s: self.tokenizer.encode(s, allowed_special={"<|endoftext|>"})
        self.decode = lambda l: self.tokenizer.decode(l)
    
    def get_train_val_split(self, text: str, split_ratio: float = 0.9) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split text into training and validation sets"""
        if not self.encode:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(split_ratio * len(data))
        
        train_data = data[:n]
        val_data = data[n:]
        
        print(f"Train set size: {len(train_data):,} tokens")
        print(f"Validation set size: {len(val_data):,} tokens")
        
        return train_data, val_data
    
    def save_vocabulary(self, path: str) -> None:
        """Save vocabulary metadata for later use"""
        vocab_data = {
            'type': 'tiktoken',
            'encoding': 'gpt2',
            'vocab_size': self.vocab_size
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary metadata saved to {path}")
    
    def load_vocabulary(self, path: str) -> None:
        """Load vocabulary from file or initialize tiktoken"""
        with open(path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            
        if vocab_data.get('type') == 'tiktoken':
            self.tokenizer = tiktoken.get_encoding(vocab_data.get('encoding', 'gpt2'))
            self.vocab_size = self.tokenizer.n_vocab
            self.encode = lambda s: self.tokenizer.encode(s, allowed_special={"<|endoftext|>"})
            self.decode = lambda l: self.tokenizer.decode(l)
            print(f"Tiktoken vocabulary loaded. Size: {self.vocab_size}")
        else:
            # Fallback to old char-level for backwards compatibility
            self.chars = vocab_data['chars']
            self.stoi = vocab_data['stoi']
            self.itos = {int(k): v for k, v in vocab_data['itos'].items()}
            self.vocab_size = len(self.chars)
            self.encode = lambda s: [self.stoi.get(c, 0) for c in s]
            self.decode = lambda l: ''.join([self.itos.get(i, '') for i in l])
            print("Legacy character vocabulary loaded.")


# Example usage and testing
if __name__ == "__main__":
    processor = DataProcessor()
    
    # Test with the existing Shakespeare data
    print("Testing with existing data...")
    try:
        text, metadata = processor.load_and_process_data("input.txt", "auto")
        print(f"Loaded text: {len(text)} characters")
        print(f"Metadata: {metadata}")
        print(f"Sample text: {text[:200]}...")
        
        # Build vocabulary
        processor.build_vocabulary(text)
        
        # Get train/val split
        train_data, val_data = processor.get_train_val_split(text)
        
        print("Data processing successful!")
        
    except Exception as e:
        print(f"Error: {e}")
