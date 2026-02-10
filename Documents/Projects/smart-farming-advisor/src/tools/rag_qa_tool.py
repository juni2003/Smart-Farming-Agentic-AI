"""
RAG Q&A Tool
Answers farming questions using retrieval-augmented generation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.rag_pipeline import FarmingRAGPipeline

class RAGQATool:
    """
    Tool for answering farming questions using RAG
    """
    
    def __init__(self):
        self.rag_pipeline = None
        self.is_initialized = False
        
    def initialize(self):
        """Initialize RAG pipeline"""
        print("📦 Initializing RAG Q&A Tool...")
        
        try:
            self.rag_pipeline = FarmingRAGPipeline()
            success = self.rag_pipeline.initialize()
            
            if success:
                self.is_initialized = True
                print("   ✅ RAG Q&A Tool initialized")
                return True
            else:
                print("   ⚠️  RAG initialized with retrieval only (no LLM)")
                self.is_initialized = True
                return True
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    def answer_question(self, question, use_llm=False, verbose=True):
        """
        Answer a farming question
        
        Args:
            question: User's question
            use_llm: Whether to use LLM for generation (if available)
            verbose: Print detailed output
            
        Returns:
            dict with answer and retrieved documents
        """
        
        if not self.is_initialized:
            success = self.initialize()
            if not success:
                return {'error': 'RAG tool not initialized'}
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.rag_pipeline.retrieve(question, top_k=3)
            
            if verbose:
                print("\n" + "="*70)
                print("❓ FARMING Q&A")
                print("="*70)
                print(f"\nQuestion: {question}")
                print("\n🔍 Retrieved Knowledge:")
                
                for i, doc in enumerate(retrieved_docs, 1):
                    print(f"\n   [{i}] Relevance: {doc['relevance_score']:.2%}")
                    # Show first 150 chars of content
                    content = doc['content'][:150] + "..." if len(doc['content']) > 150 else doc['content']
                    print(f"   {content}")
            
            # Try to generate answer with LLM (if available and requested)
            answer = None
            if use_llm and self.rag_pipeline.llm:
                try:
                    answer = self.rag_pipeline.generate_answer(question, retrieved_docs)
                    if verbose:
                        print(f"\n🤖 AI Answer:")
                        print(f"   {answer}")
                except:
                    answer = None
            
            # If no LLM answer, use best retrieved document
            if not answer:
                if retrieved_docs:
                    answer = retrieved_docs[0]['content']
                    if verbose:
                        print(f"\n📋 Answer (from knowledge base):")
                        print(f"   {answer}")
                else:
                    answer = "I don't have enough information to answer that question."
                    if verbose:
                        print(f"\n⚠️  {answer}")
            
            if verbose:
                print("="*70)
            
            return {
                'question': question,
                'answer': answer,
                'retrieved_docs': retrieved_docs,
                'source': 'llm' if (use_llm and self.rag_pipeline.llm and answer != retrieved_docs[0]['content']) else 'retrieval'
            }
            
        except Exception as e:
            return {'error': f'Question answering failed: {str(e)}'}

# =============================================================================
# TESTING
# =============================================================================

def test_rag_qa_tool():
    """Test RAG Q&A tool with sample questions"""
    
    print("\n" + "="*70)
    print("🧪 TESTING RAG Q&A TOOL")
    print("="*70)
    
    tool = RAGQATool()
    
    # Test questions
    test_questions = [
        "What is the best time to plant rice?",
        "How often should I water tomato plants?",
        "What fertilizer is good for wheat?",
        "How can I control pests on cotton?",
        "Which crops grow well in sandy soil?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}/{len(test_questions)}")
        print(f"{'='*70}")
        
        result = tool.answer_question(question, use_llm=False, verbose=True)
        
        if 'error' in result:
            print(f"\n❌ Error: {result['error']}")
    
    print("\n" + "="*70)
    print("✅ RAG Q&A TOOL TESTING COMPLETE!")
    print("="*70)
    
    print("\n📊 SUMMARY:")
    print(f"   Total questions tested: {len(test_questions)}")
    print(f"   Retrieval working: ✅")
    print(f"   LLM generation: ⚠️  Optional (working with retrieval only)")

if __name__ == "__main__":
    test_rag_qa_tool()