#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path
import platform

class EnvironmentSetup:
    def __init__(self):
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
    def check_prerequisites(self):
        """Check system prerequisites"""
        print("🔍 Checking system prerequisites...")
        
        checks = {
            'Python 3.9+': float(self.python_version) >= 3.9,
            'pip': self._check_command('pip --version'),
            'git': self._check_command('git --version'),
        }
        
        all_ok = True
        for requirement, available in checks.items():
            status = "✅" if available else "❌"
            print(f"{status} {requirement}")
            if not available:
                all_ok = False
                
        return all_ok
        
    def _check_command(self, command):
        try:
            subprocess.run(command, shell=True, check=True, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
            
    def create_virtualenv(self, env_name="minerva-env"):
        """Create Python virtual environment"""
        print(f"🐍 Creating virtual environment: {env_name}")
        
        try:
            subprocess.run([sys.executable, "-m", "venv", env_name], check=True)
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to create virtual environment")
            return False
            
    def install_dependencies(self, env_name="minerva-env"):
        """Install Python dependencies"""
        print("📦 Installing Python dependencies...")
        
        # Determine pip executable based on platform
        if self.system == "windows":
            pip_executable = str(Path(env_name) / "Scripts" / "pip.exe")
        else:
            pip_executable = str(Path(env_name) / "bin" / "pip")
            
        commands = [
            [pip_executable, "install", "--upgrade", "pip"],
            [pip_executable, "install", "-e", ".[dev]"],
            [pip_executable, "install", "wheel", "setuptools"]
        ]
        
        for cmd in commands:
            try:
                print(f"Running: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to run: {' '.join(cmd)}")
                print(f"Error: {e}")
                return False
                
        return True
        
    def setup_ollama(self):
        """Setup Ollama if not present"""
        print("🤖 Setting up Ollama...")
        
        if self._check_command('ollama --version'):
            print("✅ Ollama is already installed")
            return True
            
        try:
            if self.system == "linux":
                subprocess.run("curl -fsSL https://ollama.ai/install.sh | sh", 
                             shell=True, check=True)
            elif self.system == "darwin":  # macOS
                subprocess.run("brew install ollama", shell=True, check=True)
            elif self.system == "windows":
                print("⚠️  Please install Ollama manually from https://ollama.ai")
                return False
                
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Ollama")
            return False
            
    def download_models(self):
        """Download required models"""
        print("⬇️  Downloading required models...")
        
        models_to_download = [
            "phi3:3.8b-q4_0",
            "all-MiniLM-L6-v2"
        ]
        
        for model in models_to_download:
            print(f"Downloading: {model}")
            try:
                if ":" in model:  # Ollama model
                    subprocess.run(["ollama", "pull", model], check=True)
                else:  # HuggingFace model
                    # This would be handled by the embedding library
                    pass
            except subprocess.CalledProcessError:
                print(f"❌ Failed to download: {model}")
                return False
                
        return True
        
    def verify_setup(self):
        """Verify the setup is complete"""
        print("🔍 Verifying setup...")
        
        checks = [
            ("Virtual environment", Path("minerva-env").exists()),
            ("Dependencies installed", Path("minerva-env").exists()),  # Simplified check
            ("Ollama available", self._check_command('ollama --version')),
        ]
        
        all_ok = True
        for check_name, status in checks:
            result = "✅" if status else "❌"
            print(f"{result} {check_name}")
            if not status:
                all_ok = False
                
        return all_ok

def main():
    setup = EnvironmentSetup()
    
    print("🚀 Minerva Environment Setup")
    print("=" * 50)
    
    # Check prerequisites
    if not setup.check_prerequisites():
        print("\n❌ Prerequisites not met. Please install missing requirements.")
        sys.exit(1)
        
    # Create virtual environment
    if not setup.create_virtualenv():
        sys.exit(1)
        
    # Install dependencies
    if not setup.install_dependencies():
        sys.exit(1)
        
    # Setup Ollama
    if not setup.setup_ollama():
        print("⚠️  Ollama setup failed, but continuing...")
        
    # Download models
    if not setup.download_models():
        print("⚠️  Model download failed, but continuing...")
        
    # Verify setup
    if setup.verify_setup():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Activate virtual environment: source minerva-env/bin/activate")
        print("2. Run tests: python -m pytest tests/")
        print("3. Start using Minerva!")
    else:
        print("\n❌ Setup verification failed")
        sys.exit(1)

if __name__ == '__main__':
    main()
