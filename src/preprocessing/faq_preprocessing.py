"""
FAQ Data Preprocessing Module
Converts Q&A pairs into knowledge base for RAG
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

class FAQDataPreprocessor:
    """
    Preprocessor for FAQ Dataset to create RAG knowledge base
    """
    
    def __init__(self):
        self.df = None
        self.knowledge_base = []
        
    def load_data(self):
        """Load FAQ dataset"""
        print("📂 Loading FAQ dataset...")
        
        # Load the CSV file
        csv_path = config.FAQ_RAW_DIR / "Farming_FAQ_Assistant_Dataset.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(f"File not found: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {len(self.df)} Q&A pairs")
        print(f"   📋 Columns: {list(self.df.columns)}")
        
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*70)
        print("📊 EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        print(f"\n1️⃣ Dataset Statistics:")
        print(f"   Total rows: {len(self.df)}")
        print(f"   Columns: {list(self.df.columns)}")
        
        print(f"\n2️⃣ Missing Values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("   ✅ No missing values found!")
        else:
            for col, count in missing.items():
                if count > 0:
                    print(f"   {col}: {count}")
        
        print(f"\n3️⃣ Duplicate Analysis:")
        duplicates = self.df.duplicated().sum()
        print(f"   Total duplicate rows: {duplicates} ({duplicates/len(self.df)*100:.1f}%)")
        
        # Check unique Q&A pairs
        unique_questions = self.df['Question'].nunique()
        unique_answers = self.df['Answer'].nunique()
        print(f"   Unique questions: {unique_questions}")
        print(f"   Unique answers: {unique_answers}")
        
        print(f"\n4️⃣ Text Length Statistics:")
        self.df['question_length'] = self.df['Question'].str.len()
        self.df['answer_length'] = self.df['Answer'].str.len()
        
        print(f"\n   Question lengths:")
        print(f"      Min: {self.df['question_length'].min()}")
        print(f"      Max: {self.df['question_length'].max()}")
        print(f"      Mean: {self.df['question_length'].mean():.1f}")
        
        print(f"\n   Answer lengths:")
        print(f"      Min: {self.df['answer_length'].min()}")
        print(f"      Max: {self.df['answer_length'].max()}")
        print(f"      Mean: {self.df['answer_length'].mean():.1f}")
        
        print(f"\n5️⃣ Sample Q&A Pairs (first 5 unique):")
        unique_df = self.df.drop_duplicates(subset=['Question', 'Answer'])
        for idx, row in unique_df.head(5).iterrows():
            print(f"\n   --- Pair {idx+1} ---")
            print(f"   Q: {row['Question']}")
            print(f"   A: {row['Answer']}")
    
    def remove_duplicates(self):
        """Remove duplicate Q&A pairs"""
        print("\n🔧 Removing duplicates...")
        
        original_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['Question', 'Answer'], keep='first')
        removed = original_count - len(self.df)
        
        print(f"   Removed {removed} duplicate rows")
        print(f"   Remaining: {len(self.df)} unique Q&A pairs")
        
        return self.df
    
    def create_knowledge_base(self):
        """Create knowledge base documents for RAG"""
        print("\n📚 Creating knowledge base...")
        
        # Create natural language documents from Q&A pairs
        for idx, row in self.df.iterrows():
            question = row['Question'].strip()
            answer = row['Answer'].strip()
            
            # Format 1: Question-Answer format
            doc1 = f"Question: {question}\nAnswer: {answer}"
            
            # Format 2: Statement format (for better retrieval)
            doc2 = f"{question} {answer}"
            
            # Format 3: Direct statement
            doc3 = answer
            
            self.knowledge_base.append({
                'doc_id': f'faq_{idx}_qa',
                'content': doc1,
                'type': 'question_answer',
                'question': question,
                'answer': answer
            })
            
            self.knowledge_base.append({
                'doc_id': f'faq_{idx}_combined',
                'content': doc2,
                'type': 'combined',
                'question': question,
                'answer': answer
            })
        
        print(f"   ✅ Created {len(self.knowledge_base)} knowledge base documents")
        print(f"   (2 document formats per Q&A pair for better retrieval)")
        
        return self.knowledge_base
    
    def save_knowledge_base(self):
        """Save knowledge base to file"""
        print("\n💾 Saving knowledge base...")
        
        # Save as text file (one document per line)
        txt_path = config.FAQ_PROCESSED_PATH
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            for doc in self.knowledge_base:
                # Write content followed by separator
                f.write(doc['content'])
                f.write('\n---\n')
        
        print(f"   ✅ Saved text format: {txt_path.name}")
        
        # Save as CSV for reference
        csv_path = config.PROCESSED_DATA_DIR / "faq_knowledge_base.csv"
        kb_df = pd.DataFrame(self.knowledge_base)
        kb_df.to_csv(csv_path, index=False)
        
        print(f"   ✅ Saved CSV format: faq_knowledge_base.csv")
        
        # Save original cleaned Q&A pairs
        cleaned_path = config.PROCESSED_DATA_DIR / "faq_cleaned.csv"
        self.df[['Question', 'Answer']].to_csv(cleaned_path, index=False)
        
        print(f"   ✅ Saved cleaned Q&A: faq_cleaned.csv")
        
        # Save statistics
        stats = {
            'total_qa_pairs': len(self.df),
            'total_documents': len(self.knowledge_base),
            'unique_questions': self.df['Question'].nunique(),
            'unique_answers': self.df['Answer'].nunique(),
            'avg_question_length': float(self.df['question_length'].mean()),
            'avg_answer_length': float(self.df['answer_length'].mean())
        }
        
        import json
        stats_path = config.PROCESSED_DATA_DIR / "faq_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"   ✅ Saved statistics: faq_stats.json")
        
        print(f"\n   📁 All files saved to: {config.PROCESSED_DATA_DIR}")
    
    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("\n" + "="*70)
        print("🚀 RUNNING FAQ DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: EDA
        self.explore_data()
        
        # Step 3: Remove duplicates
        self.remove_duplicates()
        
        # Step 4: Create knowledge base
        self.create_knowledge_base()
        
        # Step 5: Save
        self.save_knowledge_base()
        
        print("\n" + "="*70)
        print("✅ FAQ DATA PREPROCESSING COMPLETE!")
        print("="*70)
        print("\n📝 Summary:")
        print(f"   • Original Q&A pairs: 500")
        print(f"   • Unique Q&A pairs: {len(self.df)}")
        print(f"   • Knowledge base documents: {len(self.knowledge_base)}")
        print(f"   • Ready for RAG pipeline!")
        
        return self

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    preprocessor = FAQDataPreprocessor()
    preprocessor.run_full_pipeline()