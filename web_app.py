#!/usr/bin/env python3
"""
VV-GPT3 Web Interface
A Flask web application for safer local GPT training and cached inference
"""

import os
import json
import threading
import time
import math
from datetime import datetime
from pathlib import Path
import torch
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_from_directory
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

# Import our VV-GPT3 modules
from src.training.data_loader import DataProcessor
from src.models.enhanced_gpt import GPT, GPTConfig
from src.services.safety import SafetyError, validate_model_name, safe_uploaded_file_path
from src.services.model_registry import (
    list_models, delete_model_family, resolve_model_for_chat, write_model_metadata
)
from src.services.training_preflight import run_preflight
from src.services.chat_cache import ChatModelCache

# Enhanced logging system
def log_activity(message, level="INFO", user_action=False):
    """Comprehensive activity logging with timestamps and emojis"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅", 
        "WARNING": "⚠️",
        "ERROR": "❌",
        "USER": "👤",
        "MODEL": "🤖",
        "UPLOAD": "📤",
        "DOWNLOAD": "📥",
        "TRAIN": "🏋️",
        "CHAT": "💬",
        "PAGE": "📄"
    }
    icon = icons.get(level, "📝")
    print(f"[{timestamp}] {icon} {message}")
    
    # Also log to file if needed
    if user_action or level in ["ERROR", "WARNING"]:
        log_file = Path('logs') / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().isoformat()}] {level}: {message}\n")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('VV_GPT3_SECRET_KEY', 'vv-gpt3-local-dev-secret')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
Path('uploads').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)
Path('logs').mkdir(exist_ok=True)

# Initialize SocketIO for real-time updates (threading mode recommended for reliability)
socketio = SocketIO(app, cors_allowed_origins=['http://127.0.0.1:5000', 'http://localhost:5000'], async_mode="threading")

chat_model_cache = ChatModelCache(max_models=2)

# Global variables for training status
training_status = {
    'is_training': False,
    'progress': 0,
    'current_loss': None,
    'best_loss': None,  # becomes a float after first evaluation
    'last_val_loss': None,
    'iteration': 0,
    'max_iterations': 0,
    'model_name': '',
    'log_messages': [],
    'start_time': None,
    'elapsed_sec': 0.0,
    'eta_sec': None,
    'iters_per_sec': None,
    'learning_rate': 0.0,
    # Control & paths
    'paused': False,
    'stopped': False,
    'latest_path': None,
    'best_path': None,
    'final_path': None
}

# Server-side event history for page persistence (replay when user navigates back)
training_event_history = []

def emit_and_record(event, data):
    """Emit a socket event AND record it for page-persistence replay."""
    socketio.emit(event, data)
    training_event_history.append({'event': event, 'data': data})

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'log'}

def get_available_models():
    """Get list of available trained models without deserializing checkpoints."""
    return list_models('models')

def get_system_info():
    """Helper to gather system information for the UI."""
    import torch
    import sys
    from pathlib import Path
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    if mps_available:
        active_device = 'mps (Apple Silicon GPU)'
    elif cuda_available:
        active_device = 'cuda'
    else:
        active_device = 'cpu'

    models_dir = Path('models')
    models_disk_bytes = sum(f.stat().st_size for f in models_dir.glob('*.pt')) if models_dir.exists() else 0

    return {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'pytorch_version': torch.__version__,
        'cuda_available': cuda_available,
        'mps_available': mps_available,
        'active_device': active_device,
        'device': 'MPS' if mps_available else ('CUDA' if cuda_available else 'CPU'),
        'device_count': torch.cuda.device_count() if cuda_available else 0,
        'models_count': len(get_available_models()),
        'models_disk_mb': models_disk_bytes / (1024 * 1024),
        'project_size': sum(f.stat().st_size for f in Path('.').rglob('*.py') if f.is_file()) / 1024 / 1024
    }

@app.route('/')
def index():
    """Main homepage"""
    client_ip = request.remote_addr
    log_activity(f"User visited homepage from {client_ip}", "PAGE", user_action=True)
    models = get_available_models()
    log_activity(f"Loaded {len(models)} available models for homepage", "MODEL")
    
    system_info = get_system_info()
    return render_template('index.html', models=models, system_info=system_info)

@app.route('/train')
def train_page():
    """Training interface page"""
    client_ip = request.remote_addr
    log_activity(f"User accessed training interface from {client_ip}", "TRAIN", user_action=True)
    return render_template('train.html')

@app.route('/chat')
def chat_page():
    """Chat interface page"""
    client_ip = request.remote_addr
    log_activity(f"User opened chat interface from {client_ip}", "CHAT", user_action=True)
    models = get_available_models()
    log_activity(f"Loaded {len(models)} models for chat interface", "MODEL")
    return render_template('chat.html', models=models)

@app.route('/models')
def models_page():
    """Model management page"""
    client_ip = request.remote_addr
    log_activity(f"User accessed model management from {client_ip}", "MODEL", user_action=True)
    models = get_available_models()
    log_activity(f"Displaying {len(models)} available models", "MODEL")
    return render_template('models.html', models=models)

@app.route('/about')
def about_page():
    """About page - comprehensive project explanation"""
    client_ip = request.remote_addr
    log_activity(f"User viewed about page from {client_ip}", "PAGE", user_action=True)
    
    system_info = get_system_info()
    log_activity("Generated system information for about page", "INFO")
    return render_template('about.html', system_info=system_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for training"""
    client_ip = request.remote_addr
    log_activity(f"File upload initiated from {client_ip}", "UPLOAD", user_action=True)
    
    if 'file' not in request.files:
        log_activity("Upload failed: No file in request", "ERROR")
        return jsonify({'success': False, 'message': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        log_activity("Upload failed: Empty filename", "ERROR")
        return jsonify({'success': False, 'message': 'No file selected'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        log_activity(f"Processing upload: {filename}", "UPLOAD")
        
        # Add timestamp to filename to avoid conflicts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext}"
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        log_activity(f"File saved as: {filename}", "SUCCESS")
        
        # Analyze the file
        try:
            log_activity(f"Analyzing uploaded file: {filename}", "INFO")
            processor = DataProcessor()
            text, metadata = processor.load_and_process_data(file_path, 'auto')
            
            return jsonify({
                'success': True, 
                'filename': filename,
                'file_path': file_path,
                'size': len(text),
                'metadata': metadata,
                'message': f'File uploaded successfully! Detected format: {metadata["format"]}'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error analyzing file: {str(e)}'})
    
    return jsonify({'success': False, 'message': 'Invalid file type. Please upload .txt files.'})

def run_training(config):
    """Run training in a separate thread"""
    global training_status
    
    try:
        print(f"🚀 Starting training with config: {config}")
        
        training_event_history.clear()
        training_status.update({
            'is_training': True,
            'progress': 0,
            'iteration': 0,
            'max_iterations': config['max_iters'],
            'model_name': config['model_name'],
            'log_messages': [],
            'current_loss': None,
            'best_loss': None,
            'last_val_loss': None,
            'start_time': time.time(),
            'elapsed_sec': 0.0,
            'eta_sec': None,
            'iters_per_sec': None,
            'learning_rate': 0.0,
            'paused': False,
            'stopped': False,
            'latest_path': None,
            'best_path': None,
            'final_path': None
        })
        
        print("📦 Importing training modules...")
        emit_and_record('training_log', {'message': '📦 Importing AI training modules...'})
        # Import training modules
        import src.training.train as train
        print("✅ Training modules imported")
        emit_and_record('training_log', {'message': '✅ AI training modules loaded successfully'})
        
        # Load and process data
        print(f"📂 Loading data from: {config['data_path']}")
        emit_and_record('training_log', {'message': '📂 Loading your training data...'})
        processor = DataProcessor()
        text, metadata = processor.load_and_process_data(config['data_path'], config['data_type'])
        print(f"✅ Data loaded: {len(text)} characters")
        emit_and_record('training_log', {'message': f'✅ Successfully loaded {len(text):,} characters of text'})
        
        # Build vocabulary
        processor.build_vocabulary(text)
        train_data, val_data = processor.get_train_val_split(text)
        print(f"📚 Vocabulary built: {processor.vocab_size} tokens")

        preflight = run_preflight(
            text, train_data, val_data,
            config['block_size'], config['batch_size'], config['model_size']
        )
        training_status['preflight'] = preflight.to_dict()
        for warning in preflight.warnings:
            emit_and_record('training_log', {'message': f'⚠️ Preflight warning: {warning}'})
        for recommendation in preflight.recommendations:
            emit_and_record('training_log', {'message': f'💡 Preflight recommendation: {recommendation}'})
        if not preflight.ok:
            raise ValueError('Preflight failed: ' + ' '.join(preflight.errors))
        
        # Configure model
        vocab_size = processor.vocab_size
        model_config = GPTConfig.get_preset(config['model_size'], vocab_size, config['block_size'])
        model_config.dropout = config['dropout']
        print(f"🧠 Model config: {model_config}")
        emit_and_record('training_log', {'message': f'🧠 Building {config["model_size"]} model with {vocab_size} vocabulary...'})
        
        # Prioritise Apple Silicon MPS, fallback to CUDA, then CPU
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        print(f"💻 Using device: {device}")
        emit_and_record('training_log', {'message': f'💻 Using {device.upper()} for training'})
        model = GPT(model_config)
        model.to(device)
        print(f"✅ Model created and moved to {device}")
        emit_and_record('training_log', {'message': '✅ Neural network model ready for training'})

        # Initialize optimizer
        train_config = {
            'learning_rate': config['learning_rate'],
            'max_iters': config['max_iters'],
            'warmup_iters': min(100, config['max_iters'] // 10),
            'lr_decay_iters': config['max_iters'],
            'min_lr': config['learning_rate'] / 10,
            'beta1': 0.9,
            'beta2': 0.95,
            'grad_clip': 1.0,
            'weight_decay': 1e-1,
        }

        optimizer = model.configure_optimizers(
            train_config['weight_decay'], 
            train_config['learning_rate'], 
            (train_config['beta1'], train_config['beta2']), 
            device.split(':')[0]
        )

        # Optional resume from latest checkpoint
        start_iter = 0
        best_val_loss = float('inf')
        latest_path_obj = Path('models') / f"{config['model_name']}_latest.pt"
        if config.get('resume', False):
            latest_path = str(latest_path_obj)
            if os.path.exists(latest_path):
                try:
                    ckpt = torch.load(latest_path, map_location=device)
                    model.load_state_dict(ckpt['model'])
                    optimizer.load_state_dict(ckpt.get('optimizer', optimizer.state_dict())) if 'optimizer' in ckpt else None
                    start_iter = int(ckpt.get('iter_num', 0)) + 1
                    best_val_loss = float(ckpt.get('val_loss', float('inf')))
                    training_status['best_loss'] = None if not math.isfinite(best_val_loss) else best_val_loss
                    training_status['latest_path'] = latest_path
                    emit_and_record('training_log', {'message': f'🔄 Resumed from latest checkpoint at step {start_iter}'})
                except Exception as e:
                    emit_and_record('training_log', {'message': f'⚠️ Failed to resume: {e}. Starting fresh.'})

        # Progress/Eval interval
        progress_interval = int(config.get('progress_interval', 50))
        eval_interval = progress_interval
        
        # Training loop with progress updates
        val_loss_history = []
        train_loss_history = []
        
        print(f"Starting training loop for {config['max_iters']} iterations...")
        emit_and_record('training_log', {'message': f'🎯 Starting training for {config["max_iters"]} iterations...'})
        emit_and_record('training_log', {'message': f'📁 Checkpoints continuously saving to models/{config["model_name"]}_[best/latest].pt'})
        
        # Timing/ETA helpers
        start_time = training_status['start_time']
        last_tick = start_time
        smoothed_ips = None
        
        best_checkpoint = None
        for iter_num in range(start_iter, config['max_iters']):
            # Update progress and timing
            progress = int((iter_num / config['max_iters']) * 100)
            now = time.time()
            elapsed = now - start_time
            instant_ips = 1.0 / max(now - last_tick, 1e-6)
            smoothed_ips = instant_ips if smoothed_ips is None else 0.9 * smoothed_ips + 0.1 * instant_ips
            last_tick = now
            remaining_iters = max(config['max_iters'] - max(iter_num, 1), 1)
            eta = remaining_iters / max(smoothed_ips, 1e-6)
            
            training_status.update({
                'progress': progress,
                'iteration': iter_num,
                'learning_rate': 0.0,  # Will be updated below
                'elapsed_sec': float(elapsed),
                'eta_sec': float(eta),
                'iters_per_sec': float(smoothed_ips)
            })
            
            # Honor pause/stop controls
            while training_status.get('paused') and not training_status.get('stopped'):
                time.sleep(0.25)
            if training_status.get('stopped'):
                emit_and_record('training_log', {'message': '🛑 Training stopped by user'})
                break

            # Learning rate scheduling
            lr = train.get_lr(iter_num, train_config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            training_status['learning_rate'] = lr
            
            # Forward pass
            X, Y = train.get_batch(train_data, config['batch_size'], config['block_size'], device)
            logits, loss = model(X, Y)
            
            # Backward pass
            loss.backward()
            
            if train_config['grad_clip'] != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Update loss
            current_loss = loss.item()
            training_status['current_loss'] = float(current_loss)
            
            # Progress updates at configured interval
            if iter_num % progress_interval == 0:
                print(f"Iteration {iter_num}, Loss: {current_loss:.4f}, LR: {lr:.6f}")
                payload = {
                    'progress': progress,
                    'iteration': iter_num,
                    'max_iterations': config['max_iters'],
                    'current_loss': float(current_loss),
                    'best_loss': float(best_val_loss) if math.isfinite(best_val_loss) else None,
                    'lr': float(lr),
                    'elapsed_sec': float(training_status['elapsed_sec']),
                    'eta_sec': float(training_status['eta_sec']) if training_status['eta_sec'] is not None else None,
                    'iters_per_sec': float(training_status['iters_per_sec']) if training_status['iters_per_sec'] is not None else None
                }
                emit_and_record('training_progress', payload)
            
            # Periodic evaluation and checkpointing at configured interval
            if iter_num % eval_interval == 0 or iter_num == config['max_iters'] - 1:
                model.eval()
                print(f"Evaluating at iteration {iter_num}...")
                val_loss = train.estimate_loss(
                    model, train_data, val_data, 10,
                    config['batch_size'], config['block_size'],
                    device, torch.no_grad()
                )['val']
                model.train()
                
                training_status['last_val_loss'] = float(val_loss)
                val_loss_history.append(float(val_loss))
                train_loss_history.append(float(current_loss))
                print(f"Validation loss: {val_loss:.4f}")
                
                # Delta calculations
                delta = 0.0
                if len(val_loss_history) >= 2:
                    delta = val_loss_history[-1] - val_loss_history[-2]

                # Overfit tripwire detection (5 consecutive increases to avoid false alarms on noisy curves)
                if len(val_loss_history) >= 6:
                    last5 = val_loss_history[-5:]
                    if all(last5[i] > last5[i-1] for i in range(1, 5)):
                        best_step_idx = val_loss_history.index(min(val_loss_history))
                        emit_and_record('training_log', {'message': '⚠️ OVERFITTING DETECTED: Validation loss has increased for 5 consecutive evaluations.'})
                        emit_and_record('training_log', {'message': f'💡 Recommendation: Your best model was saved at step {best_step_idx * eval_interval} (val_loss={min(val_loss_history):.4f}). Consider terminating training now — further iterations will degrade output quality.'})
                
                # Always save latest checkpoint
                latest_path = str(Path('models') / f"{config['model_name']}_latest.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'model_args': model_config,
                    'train_config': train_config,
                    'vocab_size': vocab_size,
                    'processor': {
                        'type': 'tiktoken',
                        'encoding': 'gpt2',
                        'vocab_size': vocab_size,
                        'chars': processor.chars,
                        'stoi': processor.stoi,
                        'itos': processor.itos
                    },
                    'iter_num': iter_num,
                    'val_loss': float(val_loss),
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, latest_path)
                training_status['latest_path'] = latest_path
                emit_and_record('checkpoint_saved', {
                    'type': 'latest',
                    'iteration': iter_num,
                    'val_loss': float(val_loss),
                    'delta': float(delta),
                    'path': latest_path
                })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    training_status['best_loss'] = float(best_val_loss)
                    best_path = str(Path('models') / f"{config['model_name']}_best.pt")
                    torch.save(checkpoint, best_path)
                    best_checkpoint = checkpoint
                    training_status['best_path'] = best_path
                    print(f"Saved best model with val_loss: {val_loss:.4f}")
                    emit_and_record('checkpoint_saved', {
                        'type': 'best',
                        'iteration': iter_num,
                        'val_loss': float(val_loss),
                        'path': best_path
                    })
                
                # Emit detailed progress update
                emit_and_record('training_progress', {
                    'progress': progress,
                    'iteration': iter_num,
                    'max_iterations': config['max_iters'],
                    'current_loss': float(current_loss),
                    'val_loss': float(val_loss),
                    'best_loss': float(best_val_loss) if math.isfinite(best_val_loss) else None,
                    'lr': float(lr),
                    'elapsed_sec': float(training_status['elapsed_sec']),
                    'eta_sec': float(training_status['eta_sec']) if training_status['eta_sec'] is not None else None,
                    'iters_per_sec': float(training_status['iters_per_sec']) if training_status['iters_per_sec'] is not None else None
                })
        
        # Training completed
        # Save final model with original name (best checkpoint if available, else latest/current)
        final_path = str(Path('models') / f"{config['model_name']}.pt")
        
        # Calculate timing metadata
        total_training_time = time.time() - start_time
        best_step = val_loss_history.index(min(val_loss_history)) * eval_interval if val_loss_history else 0
        time_to_best = (best_step / max(iter_num, 1)) * total_training_time if iter_num > 0 else 0
        
        training_metadata = {
            'model_size': config['model_size'],
            'max_iters': config['max_iters'],
            'completed_iters': iter_num,
            'learning_rate': config['learning_rate'],
            'batch_size': config.get('batch_size', 32),
            'block_size': config.get('block_size', 256),
            'dropout': config.get('dropout', 0.1),
            'best_val_loss': float(min(val_loss_history)) if val_loss_history else None,
            'best_val_loss_step': best_step,
            'final_train_loss': float(current_loss),
            'total_training_time_sec': total_training_time,
            'time_to_best_sec': time_to_best,
            'dataset_size_chars': len(text),
            'val_loss_history': val_loss_history,
            'train_loss_history': train_loss_history,
            'preflight': preflight.to_dict(),
            'checkpoint_schema_version': 3
        }
        
        if best_checkpoint is not None:
            best_checkpoint['training_metadata'] = training_metadata
            torch.save(best_checkpoint, final_path)
        elif training_status.get('latest_path') and os.path.exists(training_status['latest_path']):
            try:
                ck = torch.load(training_status['latest_path'], map_location='cpu')
                ck['training_metadata'] = training_metadata
                torch.save(ck, final_path)
            except Exception:
                # fall back to current model state
                torch.save({'model': model.state_dict(), 'model_args': model_config, 'train_config': train_config, 'vocab_size': vocab_size, 'training_metadata': training_metadata, 'processor': {'chars': processor.chars,'stoi': processor.stoi,'itos': processor.itos}}, final_path)
        else:
            torch.save({'model': model.state_dict(), 'model_args': model_config, 'train_config': train_config, 'vocab_size': vocab_size, 'training_metadata': training_metadata, 'processor': {'type': 'tiktoken', 'encoding': 'gpt2', 'vocab_size': vocab_size, 'chars': processor.chars,'stoi': processor.stoi,'itos': processor.itos}}, final_path)
        training_status['final_path'] = final_path
        try:
            write_model_metadata(config['model_name'], {
                'model_size': config['model_size'],
                'params': model.get_num_params(),
                'vocab_size': vocab_size,
                'training_metadata': training_metadata,
                'preflight': preflight.to_dict(),
                'paths': {
                    'final': final_path,
                    'best': training_status.get('best_path'),
                    'latest': training_status.get('latest_path')
                }
            }, 'models')
        except Exception as meta_exc:
            emit_and_record('training_log', {'message': f'⚠️ Metadata sidecar write failed: {meta_exc}'})

        # Generate sample text from best model
        try:
            model.eval()
            if best_checkpoint is not None:
                model.load_state_dict(best_checkpoint['model'])
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            with torch.no_grad():
                generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=50)
                generated_text = processor.decode(generated[0].tolist())
        except Exception as e:
            generated_text = f"Sample generation failed: {e}"

        training_status.update({
            'is_training': False,
            'progress': 100
        })
        
        emit_and_record('training_complete', {
            'model_name': config['model_name'],
            'final_loss': float(training_status['best_loss']) if training_status['best_loss'] is not None else None,
            'final_model_path': final_path,
            'best_model_path': training_status.get('best_path'),
            'latest_model_path': training_status.get('latest_path'),
            'elapsed_sec': training_status.get('elapsed_sec'),
            'sample_text': generated_text
        })
        
    except Exception as e:
        training_status.update({
            'is_training': False,
            'progress': 0
        })
        
        emit_and_record('training_error', {
            'error': str(e)
        })

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start model training after VV-GPT3 safety validation."""
    global training_status

    if training_status['is_training']:
        return jsonify({'success': False, 'message': 'Training already in progress'})

    data = request.json or {}

    required_fields = ['file_path', 'model_name', 'model_size', 'max_iters']
    for field in required_fields:
        if field not in data:
            return jsonify({'success': False, 'message': f'Missing field: {field}'})

    try:
        model_name = validate_model_name(data['model_name'])
        data_path = safe_uploaded_file_path(data['file_path'], app.config['UPLOAD_FOLDER'])
        if not data_path.exists():
            return jsonify({'success': False, 'message': 'Uploaded training file was not found'})

        config = {
            'data_path': str(data_path),
            'data_type': data.get('data_type', 'auto'),
            'model_name': model_name,
            'model_size': data['model_size'],
            'block_size': int(data.get('block_size', 256)),
            'batch_size': int(data.get('batch_size', 32)),
            'max_iters': int(data['max_iters']),
            'learning_rate': float(data.get('learning_rate', 3e-4)),
            'dropout': float(data.get('dropout', 0.1)),
            'progress_interval': int(data.get('progress_interval', 50)),
            'resume': bool(data.get('resume', False))
        }
    except (SafetyError, ValueError) as exc:
        return jsonify({'success': False, 'message': str(exc)})

    training_thread = threading.Thread(target=run_training, args=(config,))
    training_thread.daemon = True
    training_thread.start()

    return jsonify({'success': True, 'message': 'Training started!'})

@app.route('/training_status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/training_history')
def get_training_history():
    """Return accumulated event history for page-persistence replay."""
    return jsonify(training_event_history)

@app.route('/chat_api', methods=['POST'])
def chat_api():
    """API endpoint for cached VV-GPT3 chat generation."""
    data = request.json or {}

    try:
        model_name = validate_model_name(data.get('model_name') or data.get('model_path', ''))
        preferred_variant = data.get('variant', 'final')
        checkpoint_path = resolve_model_for_chat(model_name, preferred_variant, 'models')
        message = data['message']
        temperature = float(data.get('temperature', 0.8))
        max_length = int(data.get('max_length', 200))
        top_k = int(data.get('top_k', 50))
        top_p = float(data.get('top_p', 0.9))

        started = time.time()
        chatbot = chat_model_cache.get(checkpoint_path)
        response = chatbot.generate_response(
            message,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        elapsed = time.time() - started
        token_estimate = max(1, len(response.split()))

        return jsonify({
            'success': True,
            'response': response.strip(),
            'model_name': model_name,
            'variant': preferred_variant,
            'elapsed_sec': elapsed,
            'tokens_estimate': token_estimate,
            'tokens_per_sec_estimate': token_estimate / max(elapsed, 1e-6),
            'cache': chat_model_cache.stats()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/pause_training', methods=['POST'])
def pause_training():
    training_status['paused'] = True
    socketio.emit('training_log', {'message': '⏸️ Training paused'})
    return jsonify({'success': True})

@app.route('/resume_training', methods=['POST'])
def resume_training():
    training_status['paused'] = False
    socketio.emit('training_log', {'message': '▶️ Training resumed'})
    return jsonify({'success': True})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    training_status['stopped'] = True
    socketio.emit('training_log', {'message': '🛑 Stopping training...'})
    return jsonify({'success': True})

@app.route('/delete_model/<model_name>', methods=['POST'])
def delete_model(model_name):
    """Delete a trained model family using safe path resolution."""
    try:
        safe_name = validate_model_name(model_name)
        deleted = delete_model_family(safe_name, 'models')
        chat_model_cache.clear()
        if deleted:
            flash(f'Model {safe_name} deleted successfully', 'success')
        else:
            flash(f'Model {safe_name} not found', 'error')
    except Exception as e:
        flash(f'Error deleting model: {str(e)}', 'error')

    return redirect(url_for('models_page'))

# SocketIO event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'data': 'Connected to VV-GPT3 server'})

@socketio.on('request_training_status')
def handle_status_request():
    """Send current training status to client"""
    emit('training_status', training_status)

if __name__ == '__main__':
    print("🚀 Starting VV-GPT3 Web Interface...")
    print("🌐 Server running on: http://127.0.0.1:5000")
    print("🤖 Upload text files to train custom AI models")
    print("💬 Chat with your trained models")
    print("\nPress Ctrl+C to stop the server")
    
    # Local development mode
    socketio.run(app, debug=True, host='127.0.0.1', port=5000)
