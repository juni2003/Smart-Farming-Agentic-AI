"""
Disease Data Preprocessing Module
Handles image data loading, preprocessing, augmentation, and splitting
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import shutil
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

class DiseaseDataPreprocessor:
    """
    Preprocessor for Plant Disease Image Dataset
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.class_counts = {}
        self.image_paths = []
        self.labels = []
        
    def load_data(self):
        """Load disease dataset and organize image paths"""
        print("📂 Loading disease dataset...")
        
        # Find all image folders (classes)
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        
        # Navigate to PlantVillage folder
        plant_village_dir = config.DISEASE_RAW_DIR / "PlantVillage"
        
        if not plant_village_dir.exists():
            raise FileNotFoundError(f"PlantVillage folder not found at {plant_village_dir}")
        
        print(f"   Scanning: {plant_village_dir}")
        
        # Find all class folders
        class_folders = [d for d in plant_village_dir.iterdir() if d.is_dir()]
        
        # Remove nested PlantVillage if exists
        class_folders = [d for d in class_folders if d.name != "PlantVillage"]
        
        print(f"   Found {len(class_folders)} class folders")
        
        # Collect all image paths and labels
        for class_folder in class_folders:
            class_name = class_folder.name
            
            # Get all images in this class
            images = [f for f in class_folder.iterdir() 
                     if f.suffix in image_extensions and f.is_file()]
            
            if images:
                self.class_counts[class_name] = len(images)
                
                for img_path in images:
                    self.image_paths.append(str(img_path))
                    self.labels.append(class_name)
        
        self.class_names = sorted(list(self.class_counts.keys()))
        
        print(f"\n   ✅ Found {len(self.image_paths)} images across {len(self.class_names)} classes")
        print(f"   Classes: {self.class_names}")
        
        return self.image_paths, self.labels
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*70)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        print(f"\n1️⃣ Dataset Statistics:")
        print(f"   Total images: {len(self.image_paths)}")
        print(f"   Number of classes: {len(self.class_names)}")
        print(f"   Average images per class: {len(self.image_paths) / len(self.class_names):.1f}")
        
        print(f"\n2️⃣ Class Distribution:")
        for class_name in self.class_names:
            count = self.class_counts[class_name]
            print(f"   {class_name}: {count} images")
        
        print(f"\n3️⃣ Class Balance:")
        counts = list(self.class_counts.values())
        print(f"   Min: {min(counts)} images")
        print(f"   Max: {max(counts)} images")
        print(f"   Mean: {np.mean(counts):.1f} images")
        print(f"   Std: {np.std(counts):.1f}")
        
        # Sample image properties
        print(f"\n4️⃣ Image Properties (sampling 50 images):")
        sample_indices = np.random.choice(len(self.image_paths), min(50, len(self.image_paths)), replace=False)
        
        widths, heights, modes = [], [], []
        for idx in sample_indices:
            try:
                with Image.open(self.image_paths[idx]) as img:
                    widths.append(img.size[0])
                    heights.append(img.size[1])
                    modes.append(img.mode)
            except Exception as e:
                continue
        
        if widths:
            print(f"   Width:  Min={min(widths)}, Max={max(widths)}, Avg={np.mean(widths):.0f}")
            print(f"   Height: Min={min(heights)}, Max={max(heights)}, Avg={np.mean(heights):.0f}")
            print(f"   Modes:  {set(modes)}")
    
    def visualize_data(self, save_plots=True):
        """Create visualizations"""
        print("\n📈 Creating visualizations...")
        
        plots_dir = config.OUTPUTS_DIR / "eda_plots" / "disease"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Class distribution
        plt.figure(figsize=(14, 6))
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        plt.bar(range(len(classes)), counts, color='steelblue', edgecolor='black')
        plt.xlabel('Class Index')
        plt.ylabel('Number of Images')
        plt.title('Disease Class Distribution (15 Classes)', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(plots_dir / "class_distribution.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: class_distribution.png")
        plt.close()
        
        # 2. Class distribution with names
        plt.figure(figsize=(16, 6))
        plt.barh(classes, counts, color='steelblue', edgecolor='black')
        plt.xlabel('Number of Images')
        plt.title('Images per Disease Class', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(plots_dir / "class_distribution_detailed.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: class_distribution_detailed.png")
        plt.close()
        
        # 3. Sample images from each class
        n_classes_to_show = min(15, len(self.class_names))
        n_cols = 5
        n_rows = (n_classes_to_show + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
        axes = axes.flatten()
        
        for idx, class_name in enumerate(self.class_names[:n_classes_to_show]):
            # Get first image from this class
            class_images = [p for p, l in zip(self.image_paths, self.labels) if l == class_name]
            if class_images:
                try:
                    img = Image.open(class_images[0])
                    axes[idx].imshow(img)
                    axes[idx].set_title(f'{class_name}\n({self.class_counts[class_name]} imgs)', fontsize=8)
                    axes[idx].axis('off')
                except Exception as e:
                    axes[idx].text(0.5, 0.5, f'Error\n{class_name}', ha='center', va='center')
                    axes[idx].axis('off')
        
        # Hide extra subplots
        for idx in range(n_classes_to_show, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(plots_dir / "sample_images.png", dpi=300, bbox_inches='tight')
            print(f"   ✅ Saved: sample_images.png")
        plt.close()
        
        print(f"\n   📁 All plots saved to: {plots_dir}")
    
    def create_train_val_test_splits(self):
        """Create train/val/test splits and organize into folders"""
        print("\n🔀 Creating train/val/test splits...")
        
        # Create organized directory structure
        organized_dir = config.DISEASE_PROCESSED_DIR
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            split_dir = organized_dir / split
            split_dir.mkdir(exist_ok=True)
            for class_name in self.class_names:
                (split_dir / class_name).mkdir(exist_ok=True)
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(self.labels)
        
        print(f"   Encoded {len(self.label_encoder.classes_)} classes")
        
        # Split: 70% train, 15% val, 15% test
        indices = np.arange(len(self.image_paths))
        
        train_idx, temp_idx, train_labels, temp_labels = train_test_split(
            indices, labels_encoded,
            test_size=0.30,
            random_state=42,
            stratify=labels_encoded
        )
        
        val_idx, test_idx, val_labels, test_labels = train_test_split(
            temp_idx, temp_labels,
            test_size=0.50,
            random_state=42,
            stratify=temp_labels
        )
        
        print(f"\n   Split sizes:")
        print(f"   - Train: {len(train_idx)} images ({len(train_idx)/len(indices)*100:.1f}%)")
        print(f"   - Val:   {len(val_idx)} images ({len(val_idx)/len(indices)*100:.1f}%)")
        print(f"   - Test:  {len(test_idx)} images ({len(test_idx)/len(indices)*100:.1f}%)")
        
        # Copy images to organized structure
        print(f"\n   Organizing images into train/val/test folders...")
        
        splits = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        
        for split_name, split_indices in splits.items():
            print(f"   Processing {split_name} split...")
            for idx in tqdm(split_indices, desc=f"   {split_name}"):
                src_path = self.image_paths[idx]
                class_name = self.labels[idx]
                img_name = os.path.basename(src_path)
                dst_path = organized_dir / split_name / class_name / img_name
                
                # Copy image
                shutil.copy2(src_path, dst_path)
        
        print(f"\n   ✅ All images organized into: {organized_dir}")
        
        # Save metadata
        metadata = {
            'class_names': self.class_names,
            'class_counts': self.class_counts,
            'train_count': len(train_idx),
            'val_count': len(val_idx),
            'test_count': len(test_idx),
            'total_images': len(self.image_paths)
        }
        
        import json
        with open(organized_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Saved metadata.json")
        
        return splits
    
    def save_label_encoder(self):
        """Save label encoder"""
        print("\n💾 Saving label encoder...")
        
        joblib.dump(self.label_encoder, config.DISEASE_LABEL_ENCODER_PATH)
        print(f"   ✅ Saved: {config.DISEASE_LABEL_ENCODER_PATH.name}")
        
        # Save class names as text file for reference
        class_names_path = config.MODELS_DIR / "disease_classes.txt"
        with open(class_names_path, 'w') as f:
            for class_name in self.label_encoder.classes_:
                f.write(f"{class_name}\n")
        print(f"   ✅ Saved: disease_classes.txt")
    
    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("\n" + "="*70)
        print("🚀 RUNNING DISEASE DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: EDA
        self.explore_data()
        
        # Step 3: Visualize
        self.visualize_data(save_plots=True)
        
        # Step 4: Create splits
        self.create_train_val_test_splits()
        
        # Step 5: Save label encoder
        self.save_label_encoder()
        
        print("\n" + "="*70)
        print("✅ DISEASE DATA PREPROCESSING COMPLETE!")
        print("="*70)
        
        return self

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    preprocessor = DiseaseDataPreprocessor()
    preprocessor.run_full_pipeline()