#!/usr/bin/env python3
"""
VV-GPT3 Training Script
Train your own GPT on custom data (books, WhatsApp chats, etc.)
"""

import os
import time
import math
import pickle
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from .data_loader import DataProcessor
from ..models.enhanced_gpt import GPT, GPTConfig

def get_batch(data, batch_size, block_size, device):
    """Generate a small batch of data of inputs x and targets y"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, batch_size, block_size, device, ctx):
    """Estimate train and validation loss"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        data = train_data if split == 'train' else val_data
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, config):
    """Learning rate scheduler"""
    # 1) linear warmup for warmup_iters steps
    if it < config['warmup_iters']:
        return config['learning_rate'] * it / config['warmup_iters']
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config['lr_decay_iters']:
        return config['min_lr']
    # 3) in between, use cosine decay down to min learning rate
    if config['lr_decay_iters'] == config['warmup_iters']:
        return config['learning_rate']
    decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

def create_sample_whatsapp_data():
    """Create a sample WhatsApp conversation for testing"""
    sample_data = '''
12/07/2023, 10:30 AM - Alice: Hey! How are you doing?
12/07/2023, 10:32 AM - Bob: I'm doing great! Just finished reading this amazing book about artificial intelligence.
12/07/2023, 10:33 AM - Alice: Oh really? What was it about?
12/07/2023, 10:35 AM - Bob: It explained how neural networks work and how they can generate text that sounds human-like. It's fascinating how much progress we've made!
12/07/2023, 10:36 AM - Alice: That does sound interesting! I've been curious about AI lately too. Can you recommend any beginner-friendly resources?
12/07/2023, 10:40 AM - Bob: Definitely! There are some great online courses. The key is understanding that AI learns patterns from data, just like humans learn from experience.
12/07/2023, 10:42 AM - Alice: Makes sense! Thanks for the recommendation. I think I'll start learning about it this weekend.
12/07/2023, 10:43 AM - Bob: Awesome! Feel free to ask me if you have any questions. I love talking about this stuff 😊
12/07/2023, 10:45 AM - Alice: Will do! Thanks for being so helpful as always.
12/07/2023, 10:46 AM - Bob: Anytime! That's what friends are for.
'''
    
    with open('sample_whatsapp.txt', 'w', encoding='utf-8') as f:
        f.write(sample_data.strip())
    
    print("Created sample_whatsapp.txt for testing")

def main():
    parser = argparse.ArgumentParser(description='Train VV-GPT3')
    parser.add_argument('--data', type=str, help='Path to training data file')
    parser.add_argument('--data_type', type=str, choices=['auto', 'plain_text', 'whatsapp'], 
                        default='auto', help='Type of input data')
    parser.add_argument('--model_size', type=str, 
                        choices=['nano', 'micro', 'tiny', 'small', 'medium', 'large'], 
                        default='tiny', help='Model size preset')
    parser.add_argument('--block_size', type=int, default=256, help='Context length')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_iters', type=int, default=5000, help='Maximum training iterations')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Peak learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for faster training')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory for models')
    parser.add_argument('--eval_interval', type=int, default=250, help='Evaluation interval')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of evaluation iterations')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--create_sample', action='store_true', help='Create sample WhatsApp data')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        create_sample_whatsapp_data()
        print("Sample data created. Now run: python train.py --data sample_whatsapp.txt")
        return
    
    # Check if data argument is provided when not creating sample
    if not args.data:
        print("Error: --data argument is required for training")
        print("Use --create_sample to create sample data first")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Device setup — prioritise Apple Silicon MPS, then CUDA, then CPU
    if args.device == 'auto':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    else:
        device = args.device

    # MPS and CPU don't support torch.amp.autocast — use nullcontext for both
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    print(f"Using device: {device}")

    # Set up mixed precision training context (CUDA only; MPS uses float32 natively)
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}['float16']
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype) if device == 'cuda' else nullcontext()
    
    # Load and process data
    print(f"Loading data from {args.data}...")
    processor = DataProcessor()
    
    try:
        text, metadata = processor.load_and_process_data(args.data, args.data_type)
        print(f"Successfully loaded {len(text):,} characters")
        print(f"Data metadata: {metadata}")
        
        # Build vocabulary
        processor.build_vocabulary(text)
        
        # Get train/val split
        train_data, val_data = processor.get_train_val_split(text, split_ratio=0.9)
        
        # Save vocabulary for later use
        vocab_path = os.path.join(args.output_dir, 'vocab.json')
        processor.save_vocabulary(vocab_path)
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return
    
    # Configure model
    vocab_size = processor.vocab_size
    config = GPTConfig.get_preset(args.model_size, vocab_size, args.block_size)
    config.dropout = args.dropout
    
    print(f"Model configuration: {config}")
    
    # Create model
    model = GPT(config)
    model.to(device)
    
    # Compile model if requested (PyTorch 2.0+)
    if args.compile:
        try:
            model = torch.compile(model)
            print("Model compiled for faster training")
        except:
            print("torch.compile not available, training without compilation")
    
    # Training configuration
    train_config = {
        'learning_rate': args.learning_rate,
        'max_iters': args.max_iters,
        'warmup_iters': 100,
        'lr_decay_iters': args.max_iters,
        'min_lr': args.learning_rate / 10,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        'weight_decay': 1e-1,
    }
    
    # Initialize optimizer
    optimizer = model.configure_optimizers(train_config['weight_decay'], train_config['learning_rate'], 
                                         (train_config['beta1'], train_config['beta2']), device_type)
    if hasattr(model, 'configure_optimizers'):
        optimizer = model.configure_optimizers(train_config['weight_decay'], train_config['learning_rate'], 
                                             (train_config['beta1'], train_config['beta2']), device_type)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['learning_rate'],
                                    betas=(train_config['beta1'], train_config['beta2']),
                                    weight_decay=train_config['weight_decay'])
    
    # Training loop
    print(f"\\nStarting training for {args.max_iters} iterations...")
    print("-" * 50)
    
    best_val_loss = float('inf')
    iter_num = 0
    local_iter_num = 0
    running_mfu = -1.0
    
    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num, train_config) if train_config['lr_decay_iters'] > 0 else train_config['learning_rate']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, args.eval_iters, 
                                 args.batch_size, args.block_size, device, ctx)
            
            print(f"step {iter_num:5d}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e}")
            
            # Save checkpoint if this is the best model so far
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'train_config': train_config,
                    'vocab_size': vocab_size,
                    'processor': {
                        'type': 'tiktoken',
                        'encoding': 'gpt2',
                        'vocab_size': vocab_size,
                        'chars': processor.chars,
                        'stoi': processor.stoi,
                        'itos': processor.itos
                    }
                }
                print(f"saving checkpoint to {args.output_dir}")
                torch.save(checkpoint, os.path.join(args.output_dir, 'ckpt.pt'))
        
        # Forward backward update, with optional gradient accumulation
        if iter_num > 0 or args.eval_interval == 1:
            for micro_step in range(1):  # gradient_accumulation_steps = 1
                with ctx:
                    X, Y = get_batch(train_data, args.batch_size, args.block_size, device)
                    logits, loss = model(X, Y)
                    loss = loss / 1  # scale the loss to account for gradient accumulation
                
                # Backward pass
                loss.backward()
        else:
            # For first iteration when eval_interval > 1, create dummy loss for logging
            loss = torch.tensor(0.0)
        
        # Clip the gradient and step optimizer only if we did a forward pass
        if iter_num > 0 or args.eval_interval == 1:
            if train_config['grad_clip'] != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])
            
            # Step the optimizer
            optimizer.step()
            
            # Flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        if (iter_num % args.log_interval == 0 or iter_num == args.max_iters - 1) and hasattr(loss, 'item'):
            # Get loss as float. Note: this is a CPU-GPU sync point
            lossf = loss.item()
            print(f"iter {iter_num}: loss {lossf:.4f}")
        
        iter_num += 1
        local_iter_num += 1
        
        # Termination conditions
        if iter_num > args.max_iters:
            break
    
    print("Training completed!")
    
    # Save final model
    final_checkpoint = {
        'model': model.state_dict(),
        'model_args': config,
        'train_config': train_config,
        'vocab_size': vocab_size,
        'processor': {
            'type': 'tiktoken',
            'encoding': 'gpt2',
            'vocab_size': vocab_size,
            'chars': processor.chars,
            'stoi': processor.stoi,
            'itos': processor.itos
        }
    }
    torch.save(final_checkpoint, os.path.join(args.output_dir, 'final_model.pt'))
    print(f"Final model saved to {args.output_dir}/final_model.pt")
    
    # Test generation
    print("\\nTesting text generation...")
    model.eval()
    
    # Simple generation test
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    with torch.no_grad():
        with ctx:
            generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=50)
            generated_text = processor.decode(generated[0].tolist())
            print(f"Generated sample:\\n{generated_text}")

# Add optimizer configuration method to GPT class
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    """Configure optimizer with weight decay"""
    # Start with all of the candidate parameters
    param_dict = {pn: p for pn, p in self.named_parameters()}
    # Filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    
    return optimizer

# Monkey patch the configure_optimizers method to GPT class
import inspect
GPT.configure_optimizers = configure_optimizers

if __name__ == "__main__":
    main()
