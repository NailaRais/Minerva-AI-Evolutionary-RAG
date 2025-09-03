#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path
import subprocess

class ModelDeployer:
    def __init__(self, config_path="deploy_config.json"):
        self.config = self.load_config(config_path)
        self.models_dir = Path("models")
        self.deploy_dir = Path("deployed_models")
        
    def load_config(self, config_path):
        """Load deployment configuration"""
        default_config = {
            "target_env": "production",
            "max_model_size_gb": 3.0,
            "backup_existing": True,
            "validate_before_deploy": True
        }
        
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
        
    def validate_model(self, model_path):
        """Validate model before deployment"""
        print(f"🔍 Validating model: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # Check size
        size_gb = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file()) / (1024**3)
        if size_gb > self.config['max_model_size_gb']:
            raise ValueError(f"Model too large: {size_gb:.2f} GB > {self.config['max_model_size_gb']} GB limit")
            
        # Basic validation checks
        required_files = ['config.json', 'pytorch_model.bin', 'vocab.json']
        for file in required_files:
            if not (model_path / file).exists():
                print(f"⚠️  Warning: Missing file {file}")
                
        print(f"✅ Model validation passed: {size_gb:.2f} GB")
        return True
        
    def backup_existing(self, target_dir):
        """Backup existing deployed model"""
        if target_dir.exists() and self.config['backup_existing']:
            backup_dir = Path(f"backup_{target_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.move(target_dir, backup_dir)
            print(f"📦 Backed up existing model to: {backup_dir}")
            
    def deploy_model(self, model_name, version="latest"):
        """Deploy a model to production"""
        model_path = self.models_dir / model_name / version
        target_path = self.deploy_dir / model_name
        
        print(f"🚀 Deploying model: {model_name} (version: {version})")
        
        try:
            # Validate
            if self.config['validate_before_deploy']:
                self.validate_model(model_path)
                
            # Backup existing
            self.backup_existing(target_path)
            
            # Deploy
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                shutil.rmtree(target_path)
                
            shutil.copytree(model_path, target_path)
            
            # Create version file
            version_info = {
                "model_name": model_name,
                "version": version,
                "deployment_time": datetime.now().isoformat(),
                "size_gb": sum(f.stat().st_size for f in target_path.rglob('*') if f.is_file()) / (1024**3)
            }
            
            with open(target_path / "deployment.json", 'w') as f:
                json.dump(version_info, f, indent=2)
                
            print(f"✅ Successfully deployed {model_name} to: {target_path}")
            print(f"   Size: {version_info['size_gb']:.2f} GB")
            
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False
            
    def list_deployed_models(self):
        """List all deployed models"""
        if not self.deploy_dir.exists():
            print("No models deployed yet")
            return
            
        print("📋 Deployed Models:")
        for model_dir in self.deploy_dir.iterdir():
            if model_dir.is_dir():
                version_file = model_dir / "deployment.json"
                if version_file.exists():
                    with open(version_file, 'r') as f:
                        info = json.load(f)
                    print(f"  • {model_dir.name}: {info['version']} ({info['size_gb']:.2f} GB)")
                else:
                    print(f"  • {model_dir.name}: (no version info)")

def main():
    parser = argparse.ArgumentParser(description="Deploy Minerva models")
    parser.add_argument('action', choices=['deploy', 'list', 'validate'], help='Action to perform')
    parser.add_argument('--model', help='Model name to deploy')
    parser.add_argument('--version', default='latest', help='Model version')
    parser.add_argument('--config', default='deploy_config.json', help='Config file path')
    
    args = parser.parse_args()
    
    deployer = ModelDeployer(args.config)
    
    if args.action == 'deploy':
        if not args.model:
            print("❌ Model name required for deployment")
            return
            
        success = deployer.deploy_model(args.model, args.version)
        if success:
            print("🎉 Deployment completed successfully!")
        else:
            print("❌ Deployment failed!")
            
    elif args.action == 'list':
        deployer.list_deployed_models()
        
    elif args.action == 'validate':
        if not args.model:
            print("❌ Model name required for validation")
            return
            
        model_path = Path("models") / args.model / args.version
        try:
            deployer.validate_model(model_path)
            print("✅ Model validation passed!")
        except Exception as e:
            print(f"❌ Model validation failed: {e}")

if __name__ == '__main__':
    from datetime import datetime
    main()
