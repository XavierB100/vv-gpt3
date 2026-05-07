#!/usr/bin/env python3
"""
VV-GPT3 Desktop Launcher
Enhanced launcher with proper signal handling and comprehensive logging
"""

import os
import sys
import time
import signal
import threading
import webbrowser
from pathlib import Path
from datetime import datetime

class VVGPT3Launcher:
    def __init__(self):
        self.server_running = False
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup proper signal handling for graceful shutdown"""
        def signal_handler(sig, frame):
            print(f"\n🛑 Received signal {sig}, shutting down gracefully...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def log(self, message, level="INFO"):
        """Enhanced logging with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅", 
            "WARNING": "⚠️",
            "ERROR": "❌",
            "USER": "👤",
            "MODEL": "🤖",
            "SERVER": "🌐"
        }
        icon = icons.get(level, "📝")
        print(f"[{timestamp}] {icon} {message}")
    
    def open_browser(self):
        """Open browser after server is ready"""
        time.sleep(3)  # Wait for server to fully start
        self.log("Opening browser to http://127.0.0.1:5000", "SUCCESS")
        webbrowser.open('http://127.0.0.1:5000')
        self.log("Browser opened successfully", "SUCCESS")
    
    def shutdown(self):
        """Graceful shutdown"""
        if self.server_running:
            self.log("Shutting down VV-GPT3 server...", "WARNING")
            self.server_running = False
            time.sleep(1)
            self.log("Server stopped successfully", "SUCCESS")
        else:
            self.log("Server was not running", "INFO")
    
    def start(self):
        """Main launcher function"""
        print("\n" + "="*50)
        self.log("🚀 VV-GPT3 Desktop Application", "SUCCESS")
        print("="*50)
        
        # Change to the directory where this script is located
        script_dir = Path(__file__).parent
        os.chdir(script_dir)
        self.log(f"Working directory: {script_dir}", "INFO")
        
        # Check system info
        self.log(f"Python version: {sys.version.split()[0]}", "INFO")
        self.log(f"Platform: {sys.platform}", "INFO")
        
        # Start browser opener in background
        threading.Thread(target=self.open_browser, daemon=True).start()
        
        try:
            # Import and run the web app
            self.log("Loading Flask application...", "SERVER")
            
            # Set environment for better logging
            os.environ['FLASK_ENV'] = 'production'
            
            from web_app import app, socketio
            
            self.log("Flask application loaded successfully", "SUCCESS")
            self.log("Starting SocketIO server on http://127.0.0.1:5000", "SERVER")
            self.log("Press Ctrl+C to stop the server", "INFO")
            print("-" * 50)
            
            self.server_running = True
            
            # Start the server
            socketio.run(
                app,
                host='127.0.0.1',
                port=5000,
                debug=False,
                use_reloader=False,
                log_output=True
            )
            
        except KeyboardInterrupt:
            print()  # New line after ^C
            self.log("Keyboard interrupt received", "WARNING")
            self.shutdown()
        except Exception as e:
            self.log(f"Unexpected error: {e}", "ERROR")
            print("\nPress Enter to exit...")
            input()
        finally:
            self.log("VV-GPT3 application closed", "INFO")
            print("="*50)

def main():
    launcher = VVGPT3Launcher()
    launcher.start()

if __name__ == "__main__":
    main()
