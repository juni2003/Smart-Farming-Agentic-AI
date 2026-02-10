"""
Disease Detection Tool
Uses trained ResNet50 model to detect plant diseases from images
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

class DiseaseDetectorTool:
    """
    Tool for detecting plant diseases from images
    """
    
    def __init__(self):
        self.model = None
        self.class_names = []
        self.transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_loaded = False
        
    def load_model(self):
        """Load trained model and metadata"""
        print("📦 Loading Disease Detection Model...")
        
        try:
            # Load metadata
            metadata_path = config.MODELS_DIR / "disease_model_metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.class_names = metadata['classes']
            num_classes = len(self.class_names)
            img_size = metadata['image_size'][0]
            
            print(f"   Model: {metadata['model_name']}")
            print(f"   Classes: {num_classes}")
            print(f"   Test Accuracy: {metadata['test_accuracy']*100:.2f}%")
            
            # Build model architecture (ResNet50)
            self.model = models.resnet50(weights=None)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
            # Load weights
            model_path = config.MODELS_DIR / "disease_model_resnet50.pth"
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            self.is_loaded = True
            print(f"   ✅ Model loaded successfully (device: {self.device})")
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            return False
    
    def predict(self, image_path, top_k=3, verbose=True):
        """
        Predict disease from image
        
        Args:
            image_path: Path to plant leaf image
            top_k: Number of top predictions to return
            verbose: Print detailed output
            
        Returns:
            dict with prediction results
        """
        
        if not self.is_loaded:
            success = self.load_model()
            if not success:
                return {'error': 'Model not loaded'}
        
        try:
            # Load and preprocess image
            image_path = Path(image_path)
            if not image_path.exists():
                return {'error': f'Image not found: {image_path}'}
            
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                
                # Get top k predictions
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                predicted_class = self.class_names[top_indices[0].item()]
                confidence = top_probs[0].item()
                
                top_predictions = [
                    {
                        'disease': self.class_names[idx.item()],
                        'probability': prob.item()
                    }
                    for prob, idx in zip(top_probs, top_indices)
                ]
            
            result = {
                'predicted_disease': predicted_class,
                'confidence': confidence,
                'top_predictions': top_predictions,
                'image_path': str(image_path)
            }
            
            if verbose:
                print("\n" + "="*70)
                print("🍃 PLANT DISEASE DETECTION")
                print("="*70)
                print(f"\n📷 Image: {image_path.name}")
                print(f"   Size: {image.size}")

                print(f"\n🏆 Predicted Disease: {predicted_class}")
                print(f"   Confidence: {confidence*100:.2f}%")
                
                print(f"\n📋 Top {top_k} Predictions:")
                for i, pred in enumerate(top_predictions, 1):
                    print(f"   {i}. {pred['disease']}: {pred['probability']*100:.2f}%")
                print("="*70)
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

# =============================================================================
# TESTING
# =============================================================================

def test_disease_detector():
    """Test disease detector with sample images"""
    
    print("\n" + "="*70)
    print("🧪 TESTING DISEASE DETECTOR TOOL")
    print("="*70)
    
    tool = DiseaseDetectorTool()
    
    # Find sample test images
    test_dir = config.DISEASE_PROCESSED_DIR / 'test'
    
    if not test_dir.exists():
        print(f"\n❌ Test directory not found: {test_dir}")
        return
    
    # Get one sample image from each of first 3 classes
    sample_images = []
    for class_dir in sorted(test_dir.iterdir())[:3]:
        if class_dir.is_dir():
            images = list(class_dir.glob('*.JPG'))
            if images:
                sample_images.append({
                    'path': images[0],
                    'true_class': class_dir.name
                })
    
    if not sample_images:
        print("\n❌ No test images found")
        return
    
    print(f"\n✅ Found {len(sample_images)} test images")
    
    # Test predictions
    for i, sample in enumerate(sample_images, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}")
        print(f"True Class: {sample['true_class']}")
        print(f"{'='*70}")
        
        result = tool.predict(sample['path'], verbose=True)
        
        if 'error' not in result:
            # Check if prediction matches
            match = result['predicted_disease'] == sample['true_class']
            print(f"\n{'✅ CORRECT' if match else '❌ INCORRECT'} Prediction")
    
    print("\n" + "="*70)
    print("✅ DISEASE DETECTOR TESTING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    test_disease_detector()