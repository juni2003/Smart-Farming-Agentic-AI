"""
Disease Detection Deep Learning Model Training (PyTorch)
Trains CNN and Transfer Learning models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import time
import json
from PIL import Image

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class PlantDiseaseDataset(Dataset):
    """Custom Dataset for Plant Disease Images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # Collect all image paths
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*.JPG'):
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class SimpleCNN(nn.Module):
    """Simple CNN Model - CORRECTED"""
    
    def __init__(self, num_classes, img_size=224):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
        )
        
        # Calculate flattened size: 256 channels * 14 * 14 = 50176
        # (224 / 2^4 = 14)
        self.feature_size = 256 * (img_size // 16) * (img_size // 16)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DiseaseModelTrainer:
    """Trains CNN and Transfer Learning models"""
    
    def __init__(self, quick_test=False):
        self.quick_test = quick_test
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.class_names = None
        
        self.img_size = config.DISEASE_MODEL_CONFIG['image_size'][0]
        self.batch_size = 16 if quick_test else config.DISEASE_MODEL_CONFIG['batch_size']
        self.epochs = 5 if quick_test else config.DISEASE_MODEL_CONFIG['epochs']
        
        if quick_test:
            print("⚠️  QUICK TEST MODE: Training on reduced dataset")
    
    def setup_data_loaders(self):
        """Setup data loaders with augmentation"""
        print("📂 Setting up data loaders...")
        
        # Data paths
        train_dir = config.DISEASE_PROCESSED_DIR / 'train'
        val_dir = config.DISEASE_PROCESSED_DIR / 'val'
        test_dir = config.DISEASE_PROCESSED_DIR / 'test'
        
        # Transforms
        train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Datasets
        train_dataset = PlantDiseaseDataset(train_dir, train_transform)
        val_dataset = PlantDiseaseDataset(val_dir, test_transform)
        test_dataset = PlantDiseaseDataset(test_dir, test_transform)
        
        self.class_names = train_dataset.class_names
        
        # DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                      shuffle=True, num_workers=0)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                    shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, 
                                     shuffle=False, num_workers=0)
        
        print(f"   ✅ Train samples: {len(train_dataset)}")
        print(f"   ✅ Val samples: {len(val_dataset)}")
        print(f"   ✅ Test samples: {len(test_dataset)}")
        print(f"   ✅ Classes: {len(self.class_names)}")
        print(f"   ✅ Batch size: {self.batch_size}")
    
    def train_model(self, model, model_name):
        """Train a model"""
        print(f"\n🏋️  Training {model_name}...")
        print("="*70)
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({'loss': f'{train_loss/(pbar.n+1):.4f}',
                                'acc': f'{100.*train_correct/train_total:.2f}%'})
            
            train_loss /= len(self.train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in self.val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_loss /= len(self.val_loader)
            val_acc = 100. * val_correct / val_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f'   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    print(f'   Early stopping at epoch {epoch+1}')
                    break
        
        model.load_state_dict(best_model_state)
        train_time = time.time() - start_time
        
        print(f"\n   ✅ Training completed in {train_time/60:.2f} minutes")
        print(f"   Best Val Accuracy: {best_val_acc:.2f}%")
        
        return model, history, train_time
    
    def evaluate_model(self, model, model_name):
        """Evaluate model on test set"""
        print(f"\n🧪 Evaluating {model_name} on test set...")
        
        model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images = images.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                
                y_pred.extend(predicted.cpu().numpy())
                y_true.extend(labels.numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n   Test Accuracy: {accuracy:.4f}")
        print(f"\n   📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names, zero_division=0))
        
        return np.array(y_pred), np.array(y_true), accuracy
    
    def visualize_training(self, history, model_name):
        """Visualize training curves"""
        results_dir = config.OUTPUTS_DIR / "results" / "disease"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs_range = range(1, len(history['train_acc']) + 1)
        
        ax1.plot(epochs_range, history['train_acc'], label='Train Accuracy', linewidth=2)
        ax1.plot(epochs_range, history['val_acc'], label='Val Accuracy', linewidth=2)
        ax1.set_title(f'{model_name} - Accuracy', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        ax2.plot(epochs_range, history['train_loss'], label='Train Loss', linewidth=2)
        ax2.plot(epochs_range, history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / f"{model_name.replace(' ', '_')}_training_curves.png", 
                   dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {model_name}_training_curves.png")
        plt.close()
    
    def visualize_confusion_matrix(self, y_true, y_pred, model_name):
        """Visualize confusion matrix"""
        results_dir = config.OUTPUTS_DIR / "results" / "disease"
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(results_dir / f"{model_name.replace(' ', '_')}_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {model_name}_confusion_matrix.png")
        plt.close()
    
    def save_best_model(self):
        """Save the best model"""
        print("\n💾 Saving best model...")
        
        torch.save(self.best_model.state_dict(), config.DISEASE_MODEL_PATH.with_suffix('.pth'))
        print(f"   ✅ Saved: {config.DISEASE_MODEL_PATH.with_suffix('.pth').name}")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'test_accuracy': float(self.results[self.best_model_name]['accuracy']),
            'train_time': float(self.results[self.best_model_name]['train_time']),
            'num_classes': len(self.class_names),
            'classes': self.class_names,
            'image_size': [self.img_size, self.img_size]
        }
        
        metadata_path = config.MODELS_DIR / "disease_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Saved: disease_model_metadata.json")
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("\n" + "="*70)
        print("🚀 RUNNING DISEASE MODEL TRAINING PIPELINE (PyTorch)")
        print("="*70)
        
        # Setup data
        self.setup_data_loaders()
        
        # Train Simple CNN
        print("\n" + "="*70)
        print("MODEL 1: SIMPLE CNN")
        print("="*70)
        cnn_model = SimpleCNN(len(self.class_names), self.img_size)
        cnn_model, cnn_history, cnn_time = self.train_model(cnn_model, "Simple CNN")
        cnn_pred, cnn_true, cnn_acc = self.evaluate_model(cnn_model, "Simple CNN")
        self.visualize_training(cnn_history, "Simple CNN")
        self.visualize_confusion_matrix(cnn_true, cnn_pred, "Simple CNN")
        
        self.models['Simple CNN'] = cnn_model
        self.results['Simple CNN'] = {
            'accuracy': cnn_acc,
            'train_time': cnn_time,
            'y_pred': cnn_pred,
            'y_true': cnn_true
        }
        
        # Train ResNet50
        print("\n" + "="*70)
        print("MODEL 2: RESNET50")
        print("="*70)
        resnet_model = models.resnet50(weights='IMAGENET1K_V1')  # Updated for newer PyTorch
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(self.class_names))
        resnet_model, resnet_history, resnet_time = self.train_model(resnet_model, "ResNet50")
        resnet_pred, resnet_true, resnet_acc = self.evaluate_model(resnet_model, "ResNet50")
        self.visualize_training(resnet_history, "ResNet50")
        self.visualize_confusion_matrix(resnet_true, resnet_pred, "ResNet50")
        
        self.models['ResNet50'] = resnet_model
        self.results['ResNet50'] = {
            'accuracy': resnet_acc,
            'train_time': resnet_time,
            'y_pred': resnet_pred,
            'y_true': resnet_true
        }
        
        # Compare
        print("\n" + "="*70)
        print("📈 MODEL COMPARISON")
        print("="*70)
        
        comparison = []
        for name, result in self.results.items():
            comparison.append({
                'Model': name,
                'Test Accuracy': f"{result['accuracy']:.4f}",
                'Train Time (min)': f"{result['train_time']/60:.2f}"
            })
        
        comp_df = pd.DataFrame(comparison)
        print(comp_df.to_string(index=False))
        
        self.best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"   Test Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        
        # Save
        self.save_best_model()
        
        print("\n" + "="*70)
        print("✅ DISEASE MODEL TRAINING COMPLETE!")
        print("="*70)
        
        return self

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick-test', action='store_true')
    args = parser.parse_args()
    
    trainer = DiseaseModelTrainer(quick_test=args.quick_test)
    trainer.run_full_pipeline()