"""
RAG Pipeline for Farming Q&A
Retrieval-Augmented Generation using FAISS + Google Gemini
"""

import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import joblib
import json
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

class FarmingRAGPipeline:
    """
    RAG Pipeline for farming questions
    """
    
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.llm = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize all components"""
        print("🚀 Initializing RAG Pipeline...")
        
        # 1. Load embedding model
        print("\n📊 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"   ✅ Loaded: all-MiniLM-L6-v2")
        
        # 2. Load or create vector store
        self.load_or_create_vector_store()
        
        # 3. Initialize LLM
        print("\n🤖 Initializing Google Gemini LLM...")
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.llm = genai.GenerativeModel('gemini-2.5-flash')
            print("   ✅ Gemini Pro initialized")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print("   ⚠️  Make sure GOOGLE_API_KEY is set in .env file")
            return False
        
        self.is_initialized = True
        print("\n✅ RAG Pipeline initialized successfully!")
        return True
    
    def load_or_create_vector_store(self):
        """Load existing vector store or create new one"""
        
        vector_store_path = config.MODELS_DIR / "faq_vector_store.index"
        documents_path = config.MODELS_DIR / "faq_documents.json"
        
        if vector_store_path.exists() and documents_path.exists():
            print("\n📂 Loading existing vector store...")
            self.index = faiss.read_index(str(vector_store_path))
            with open(documents_path, 'r') as f:
                self.documents = json.load(f)
            print(f"   ✅ Loaded {len(self.documents)} documents")
        else:
            print("\n🔨 Creating new vector store...")
            self.create_vector_store()
    
    def create_vector_store(self):
        """Create vector store from FAQ knowledge base"""
        
        # Load knowledge base
        kb_path = config.FAQ_PROCESSED_PATH
        
        if not kb_path.exists():
            print(f"   ❌ Knowledge base not found: {kb_path}")
            print("   Please run FAQ preprocessing first!")
            return
        
        # Read documents
        print(f"   Reading: {kb_path}")
        with open(kb_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by separator
        raw_docs = content.split('\n---\n')
        self.documents = [doc.strip() for doc in raw_docs if doc.strip()]
        
        print(f"   Found {len(self.documents)} documents")
        
        # Generate embeddings
        print("   Generating embeddings...")
        embeddings = self.embedding_model.encode(
            self.documents,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"   ✅ Created index with {self.index.ntotal} vectors")
        
        # Save
        print("   Saving vector store...")
        vector_store_path = config.MODELS_DIR / "faq_vector_store.index"
        documents_path = config.MODELS_DIR / "faq_documents.json"
        
        faiss.write_index(self.index, str(vector_store_path))
        with open(documents_path, 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        print(f"   ✅ Saved to: {config.MODELS_DIR}")
    
    def retrieve(self, query, top_k=3):
        """Retrieve most relevant documents"""
        
        if not self.is_initialized:
            print("❌ Pipeline not initialized. Call initialize() first.")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Get documents
        retrieved_docs = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                retrieved_docs.append({
                    'content': self.documents[idx],
                    'distance': float(distance),
                    'relevance_score': float(1 / (1 + distance))
                })
        
        return retrieved_docs
    
    def generate_answer(self, query, context_docs):
        """Generate answer using LLM"""
        
        if not self.is_initialized or not self.llm:
            return "Error: LLM not initialized"
        
        # Prepare context
        context = "\n\n".join([doc['content'] for doc in context_docs])
        
        # Create prompt
        prompt = f"""You are a helpful farming advisor AI assistant. Answer the user's question based on the provided context.

Context (Farming Knowledge):
{context}

User Question: {query}

Instructions:
- Provide a clear, concise, and helpful answer
- Use the context information if relevant
- If the context doesn't fully answer the question, use your general knowledge about farming
- Be practical and actionable
- If you don't know, say so honestly

Answer:"""
        
        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def query(self, question, top_k=3, verbose=False):
        """
        Complete RAG query: retrieve + generate
        """
        
        if not self.is_initialized:
            return {
                'error': 'Pipeline not initialized',
                'answer': None
            }
        
        if verbose:
            print(f"\n❓ Question: {question}")
            print("="*70)
        
        # Step 1: Retrieve
        if verbose:
            print("\n🔍 Retrieving relevant documents...")
        
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        if verbose:
            print(f"   Found {len(retrieved_docs)} documents")
            for i, doc in enumerate(retrieved_docs, 1):
                print(f"\n   Document {i} (relevance: {doc['relevance_score']:.4f}):")
                print(f"   {doc['content'][:200]}...")
        
        # Step 2: Generate
        if verbose:
            print("\n🤖 Generating answer with Gemini Pro...")
        
        answer = self.generate_answer(question, retrieved_docs)
        
        if verbose:
            print(f"\n✅ Answer:\n{answer}")
            print("="*70)
        
        return {
            'question': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'num_docs_retrieved': len(retrieved_docs)
        }
    
    def batch_query(self, questions, verbose=False):
        """Process multiple questions"""
        results = []
        for question in questions:
            result = self.query(question, verbose=verbose)
            results.append(result)
        return results

# =============================================================================
# TESTING FUNCTION
# =============================================================================

def test_rag_pipeline():
    """Test the RAG pipeline with sample questions"""
    
    print("\n" + "="*70)
    print("🧪 TESTING RAG PIPELINE")
    print("="*70)
    
    # Initialize
    rag = FarmingRAGPipeline()
    success = rag.initialize()
    
    if not success:
        print("\n❌ Initialization failed. Check your GOOGLE_API_KEY.")
        return
    
    # Test questions
    test_questions = [
        "What is the best time to plant rice?",
        "How often should I water tomato plants?",
        "What fertilizer is good for wheat?",
        "How can I control pests on cotton?",
        "Which crops grow well in sandy soil?",
        "What are the symptoms of bacterial spot in peppers?",  # Not in KB
        "How do I increase crop yield?"  # General question
    ]
    
    print("\n" + "="*70)
    print("📝 RUNNING TEST QUERIES")
    print("="*70)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{len(test_questions)}")
        result = rag.query(question, verbose=True)
        print()
    
    print("\n" + "="*70)
    print("✅ RAG PIPELINE TESTING COMPLETE!")
    print("="*70)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    test_rag_pipeline()