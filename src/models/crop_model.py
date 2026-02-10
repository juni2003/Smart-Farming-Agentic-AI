"""
Crop Recommendation ML Model Training
Trains multiple models and selects the best one
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

class CropModelTrainer:
    """
    Trains and evaluates ML models for crop recommendation
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.label_encoder = None
        self.scaler = None
        
    def load_data(self):
        """Load preprocessed data"""
        print("📂 Loading preprocessed data...")
        
        splits_dir = config.PROCESSED_DATA_DIR / "crop_splits"
        
        self.X_train = np.load(splits_dir / "X_train.npy")
        self.X_val = np.load(splits_dir / "X_val.npy")
        self.X_test = np.load(splits_dir / "X_test.npy")
        self.y_train = np.load(splits_dir / "y_train.npy")
        self.y_val = np.load(splits_dir / "y_val.npy")
        self.y_test = np.load(splits_dir / "y_test.npy")
        
        # Load label encoder and scaler
        self.label_encoder = joblib.load(config.CROP_LABEL_ENCODER_PATH)
        self.scaler = joblib.load(config.CROP_SCALER_PATH)
        
        print(f"   ✅ Train: {self.X_train.shape}")
        print(f"   ✅ Val:   {self.X_val.shape}")
        print(f"   ✅ Test:  {self.X_test.shape}")
        print(f"   ✅ Classes: {len(self.label_encoder.classes_)}")
        
        return self.X_train, self.y_train
    
    def initialize_models(self):
        """Initialize ML models"""
        print("\n🤖 Initializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial',
                solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        }
        
        print(f"   ✅ Initialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"      • {name}")
    
    def train_models(self):
        """Train all models"""
        print("\n🏋️ Training models...")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\n📊 Training: {name}")
            print("-"*70)
            
            start_time = time.time()
            
            # Train
            model.fit(self.X_train, self.y_train)
            
            train_time = time.time() - start_time
            
            # Predict on validation set
            y_val_pred = model.predict(self.X_val)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_val, y_val_pred)
            precision = precision_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
            recall = recall_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
            f1 = f1_score(self.y_val, y_val_pred, average='weighted', zero_division=0)
            
            # Store results
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'train_time': train_time,
                'y_val_pred': y_val_pred
            }
            
            print(f"   Training time: {train_time:.2f} seconds")
            print(f"   Validation Accuracy:  {accuracy:.4f}")
            print(f"   Validation Precision: {precision:.4f}")
            print(f"   Validation Recall:    {recall:.4f}")
            print(f"   Validation F1-Score:  {f1:.4f}")
        
        print("\n" + "="*70)
    
    def evaluate_models(self):
        """Compare and select best model"""
        print("\n📈 Model Comparison:")
        print("="*70)
        
        # Create comparison dataframe
        comparison = []
        for name, result in self.results.items():
            comparison.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'Train Time (s)': result['train_time']
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        print(comparison_df.to_string(index=False))
        print("="*70)
        
        # Select best model based on F1-score
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        self.best_model = self.results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {self.results[best_model_name]['f1_score']:.4f}")
        
        return self.best_model
    
    def evaluate_on_test_set(self):
        """Evaluate best model on test set"""
        print("\n🧪 Evaluating on Test Set...")
        print("="*70)
        
        # Predict on test set
        y_test_pred = self.best_model.predict(self.X_test)
        
        # Calculate metrics
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted', zero_division=0)
        
        print(f"\n{self.best_model_name} - Test Set Performance:")
        print(f"   Accuracy:  {test_accuracy:.4f}")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall:    {test_recall:.4f}")
        print(f"   F1-Score:  {test_f1:.4f}")
        
        # Classification report
        print(f"\n📋 Classification Report:")
        print("-"*70)
        print(classification_report(
            self.y_test, 
            y_test_pred, 
            target_names=self.label_encoder.classes_,
            zero_division=0
        ))
        
        return y_test_pred, test_accuracy
    
    def visualize_results(self, y_test_pred):
        """Create visualizations"""
        print("\n📊 Creating visualizations...")
        
        results_dir = config.OUTPUTS_DIR / "results" / "crop"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Model comparison bar plot
        plt.figure(figsize=(12, 6))
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score']
            })
        
        df_comp = pd.DataFrame(comparison_data)
        df_comp = df_comp.set_index('Model')
        
        df_comp.plot(kind='bar', width=0.8, edgecolor='black')
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='lower right')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: model_comparison.png")
        plt.close()
        
        # 2. Confusion Matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.best_model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(results_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: confusion_matrix.png")
        plt.close()
        
        # 3. Feature importance (if Random Forest or XGBoost)
        if self.best_model_name in ['Random Forest', 'XGBoost']:
            feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 
                           'rainfall', 'soil_fertility', 'climate_index', 
                           'npk_ratio', 'moisture_index']
            
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(importances)), importances[indices], edgecolor='black')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.title(f'Feature Importance - {self.best_model_name}', fontsize=14, fontweight='bold')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(results_dir / "feature_importance.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: feature_importance.png")
            plt.close()
        
        print(f"\n   📁 All plots saved to: {results_dir}")
    
    def save_model(self):
        """Save the best model"""
        print("\n💾 Saving best model...")
        
        joblib.dump(self.best_model, config.CROP_MODEL_PATH)
        print(f"   ✅ Saved: {config.CROP_MODEL_PATH.name}")
        
        # Save model metadata
        metadata = {
            'model_name': self.best_model_name,
            'accuracy': float(self.results[self.best_model_name]['accuracy']),
            'precision': float(self.results[self.best_model_name]['precision']),
            'recall': float(self.results[self.best_model_name]['recall']),
            'f1_score': float(self.results[self.best_model_name]['f1_score']),
            'train_time': float(self.results[self.best_model_name]['train_time']),
            'num_classes': len(self.label_encoder.classes_),
            'classes': list(self.label_encoder.classes_),
            'num_features': self.X_train.shape[1]
        }
        
        import json
        metadata_path = config.MODELS_DIR / "crop_model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Saved: crop_model_metadata.json")
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        print("\n" + "="*70)
        print("🚀 RUNNING CROP MODEL TRAINING PIPELINE")
        print("="*70)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Initialize models
        self.initialize_models()
        
        # Step 3: Train models
        self.train_models()
        
        # Step 4: Evaluate and compare
        self.evaluate_models()
        
        # Step 5: Test best model
        y_test_pred, test_accuracy = self.evaluate_on_test_set()
        
        # Step 6: Visualize
        self.visualize_results(y_test_pred)
        
        # Step 7: Save
        self.save_model()
        
        print("\n" + "="*70)
        print("✅ CROP MODEL TRAINING COMPLETE!")
        print("="*70)
        print(f"\n🏆 Best Model: {self.best_model_name}")
        print(f"📊 Test Accuracy: {test_accuracy:.4f}")
        print(f"💾 Model saved to: {config.CROP_MODEL_PATH}")
        
        return self

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    trainer = CropModelTrainer()
    trainer.run_full_pipeline()