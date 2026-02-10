"""
Crop Data Preprocessing Module
Handles data loading, cleaning, feature engineering, and splitting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

class CropDataPreprocessor:
    """
    Preprocessor for Crop Recommendation Dataset
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
    def load_data(self):
        """Load crop dataset"""
        print("📂 Loading crop dataset...")
        
        # Load the CSV file
        csv_path = config.CROP_RAW_DIR / "Crop_recommendation.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"   📋 Columns: {list(self.df.columns)}")
        
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*70)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        # Basic info
        print("\n1️⃣ Dataset Shape:")
        print(f"   Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        print("\n2️⃣ Column Names and Types:")
        for col, dtype in self.df.dtypes.items():
            print(f"   {col}: {dtype}")
        
        print("\n3️⃣ Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   ✅ No missing values found!")
        else:
            for col, count in missing.items():
                if count > 0:
                    print(f"   {col}: {count}")
        
        print("\n4️⃣ Target Variable (label):")
        print(f"   Number of unique crops: {self.df['label'].nunique()}")
        print(f"   Class distribution:")
        print(self.df['label'].value_counts())
        
        print("\n5️⃣ Statistical Summary:")
        print(self.df.describe())
        
        return self.df.describe()
    
    def visualize_data(self, save_plots=True):
        """Create visualizations for EDA"""
        print("\n📈 Creating visualizations...")
        
        # Create output directory for plots
        plots_dir = config.OUTPUTS_DIR / "eda_plots" / "crop"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature distributions
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('Crop Features Distribution', fontsize=16, fontweight='bold')
        
        features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        for idx, feature in enumerate(features):
            row = idx // 3
            col = idx % 3
            axes[row, col].hist(self.df[feature], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(alpha=0.3)
        
        # Hide empty subplot
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(plots_dir / "feature_distributions.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: feature_distributions.png")
        plt.close()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.df[features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.2f')
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig(plots_dir / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: correlation_heatmap.png")
        plt.close()
        
        # 3. Crop distribution
        plt.figure(figsize=(14, 6))
        crop_counts = self.df['label'].value_counts()
        crop_counts.plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title('Crop Distribution (22 Classes)', fontsize=16, fontweight='bold')
        plt.xlabel('Crop Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig(plots_dir / "crop_distribution.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: crop_distribution.png")
        plt.close()
        
        # 4. Box plots for outlier detection
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('Box Plots for Outlier Detection', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(features):
            row = idx // 4
            col = idx % 4
            axes[row, col].boxplot(self.df[feature])
            axes[row, col].set_title(feature)
            axes[row, col].set_ylabel('Value')
            axes[row, col].grid(alpha=0.3)
        
        axes[1, 3].axis('off')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(plots_dir / "outlier_detection.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: outlier_detection.png")
        plt.close()
        
        print(f"\n   📁 All plots saved to: {plots_dir}")
    
    def feature_engineering(self):
        """Create new engineered features"""
        print("\n🔧 Performing feature engineering...")
        
        # 1. Soil fertility index (NPK average)
        self.df['soil_fertility'] = (self.df['N'] + self.df['P'] + self.df['K']) / 3
        print("   ✅ Created: soil_fertility = (N + P + K) / 3")
        
        # 2. Climate index
        self.df['climate_index'] = self.df['temperature'] * self.df['humidity'] / 100
        print("   ✅ Created: climate_index = temperature * humidity / 100")
        
        # 3. NPK ratio
        self.df['npk_ratio'] = self.df['N'] / (self.df['P'] + self.df['K'] + 1e-5)
        print("   ✅ Created: npk_ratio = N / (P + K)")
        
        # 4. Moisture index
        self.df['moisture_index'] = self.df['rainfall'] * self.df['humidity'] / 100
        print("   ✅ Created: moisture_index = rainfall * humidity / 100")
        
        print(f"\n   Total features now: {len(self.df.columns) - 1}")  # -1 for label
        
        return self.df
    
    def preprocess_and_split(self):
        """Preprocess data and split into train/val/test"""
        print("\n🔀 Splitting and preprocessing data...")
        
        # Separate features and target
        feature_cols = [col for col in self.df.columns if col != 'label']
        X = self.df[feature_cols]
        y = self.df['label']
        
        print(f"   Features: {len(feature_cols)}")
        print(f"   Feature names: {feature_cols}")
        print(f"   Samples: {len(X)}")
        print(f"   Classes: {y.nunique()}")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"\n   ✅ Encoded {len(self.label_encoder.classes_)} crop labels")
        print(f"   Classes: {list(self.label_encoder.classes_)}")
        
        # Split: 70% train, 15% val, 15% test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y_encoded, 
            test_size=config.CROP_MODEL_CONFIG['test_size'],
            random_state=config.CROP_MODEL_CONFIG['random_state'],
            stratify=y_encoded
        )
        
        val_size_adjusted = config.CROP_MODEL_CONFIG['val_size'] / (1 - config.CROP_MODEL_CONFIG['test_size'])
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=config.CROP_MODEL_CONFIG['random_state'],
            stratify=y_temp
        )
        
        print(f"\n   Split sizes:")
        print(f"   - Training:   {len(self.X_train)} ({len(self.X_train)/len(X)*100:.1f}%)")
        print(f"   - Validation: {len(self.X_val)} ({len(self.X_val)/len(X)*100:.1f}%)")
        print(f"   - Test:       {len(self.X_test)} ({len(self.X_test)/len(X)*100:.1f}%)")
        
        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=feature_cols
        )
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=feature_cols
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=feature_cols
        )
        
        print(f"\n   ✅ Applied StandardScaler normalization")
        
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def save_preprocessed_data(self):
        """Save preprocessed data and objects"""
        print("\n💾 Saving preprocessed data and objects...")
        
        # Save processed dataframe
        self.df.to_csv(config.CROP_PROCESSED_PATH, index=False)
        print(f"   ✅ Saved processed data: {config.CROP_PROCESSED_PATH.name}")
        
        # Save scaler
        joblib.dump(self.scaler, config.CROP_SCALER_PATH)
        print(f"   ✅ Saved scaler: {config.CROP_SCALER_PATH.name}")
        
        # Save label encoder
        joblib.dump(self.label_encoder, config.CROP_LABEL_ENCODER_PATH)
        print(f"   ✅ Saved label encoder: {config.CROP_LABEL_ENCODER_PATH.name}")
        
        # Save splits as numpy arrays
        splits_dir = config.PROCESSED_DATA_DIR / "crop_splits"
        splits_dir.mkdir(exist_ok=True)
        
        np.save(splits_dir / "X_train.npy", self.X_train.values)
        np.save(splits_dir / "X_val.npy", self.X_val.values)
        np.save(splits_dir / "X_test.npy", self.X_test.values)
        np.save(splits_dir / "y_train.npy", self.y_train)
        np.save(splits_dir / "y_val.npy", self.y_val)
        np.save(splits_dir / "y_test.npy", self.y_test)
        
        print(f"   ✅ Saved train/val/test splits")
        print(f"\n   📁 All files saved to: {config.PROCESSED_DATA_DIR}")
    
    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("\n" + "="*70)
        print("🚀 RUNNING CROP DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: EDA
        self.explore_data()
        
        # Step 3: Visualize
        self.visualize_data(save_plots=True)
        
        # Step 4: Feature engineering
        self.feature_engineering()
        
        # Step 5: Preprocess and split
        self.preprocess_and_split()
        
        # Step 6: Save
        self.save_preprocessed_data()
        
        print("\n" + "="*70)
        print("✅ CROP DATA PREPROCESSING COMPLETE!")
        print("="*70)
        
        return self

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    preprocessor = CropDataPreprocessor()
    preprocessor.run_full_pipeline()