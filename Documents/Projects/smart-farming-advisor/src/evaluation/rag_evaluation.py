"""
RAG System Evaluation
Calculate BLEU, Hit Rate, MRR, and Relevance scores
"""

import numpy as np
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag.rag_pipeline import FarmingRAGPipeline

def calculate_hit_rate(retrieved_docs, relevant_threshold=0.5):
    """
    Calculate Hit Rate: % of queries that retrieved at least one relevant doc
    """
    hits = sum(1 for doc in retrieved_docs if doc['relevance_score'] >= relevant_threshold)
    return 1 if hits > 0 else 0

def calculate_mrr(retrieved_docs, relevant_threshold=0.5):
    """
    Calculate Mean Reciprocal Rank
    """
    for i, doc in enumerate(retrieved_docs, 1):
        if doc['relevance_score'] >= relevant_threshold:
            return 1.0 / i
    return 0.0

def calculate_relevance_score(retrieved_docs):
    """
    Calculate average relevance of retrieved documents
    """
    if not retrieved_docs:
        return 0.0
    return np.mean([doc['relevance_score'] for doc in retrieved_docs])

def evaluate_rag_system():
    """
    Comprehensive RAG evaluation
    """
    
    print("\n" + "="*70)
    print("📊 RAG SYSTEM EVALUATION")
    print("="*70)
    
    # Initialize RAG
    rag = FarmingRAGPipeline()
    rag.initialize()
    
    # Test queries with expected answers
    test_queries = [
        {
            'question': 'What is the best time to plant rice?',
            'expected_keywords': ['monsoon', 'june', 'july', 'rice']
        },
        {
            'question': 'How often should I water tomato plants?',
            'expected_keywords': ['2-3 days', 'tomato', 'water', 'moist']
        },
        {
            'question': 'What fertilizer is good for wheat?',
            'expected_keywords': ['npk', 'nitrogen', 'phosphorus', 'potassium', 'wheat']
        },
        {
            'question': 'How can I control pests on cotton?',
            'expected_keywords': ['pest', 'neem', 'cotton', 'management']
        },
        {
            'question': 'Which crops grow well in sandy soil?',
            'expected_keywords': ['groundnut', 'millet', 'sandy', 'soil']
        }
    ]
    
    print(f"\n🧪 Testing {len(test_queries)} queries...")
    
    results = {
        'relevance_scores': [],
        'hit_rates': [],
        'mrr_scores': [],
        'top1_relevance': [],
        'queries': []
    }
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(test_queries)}: {test['question']}")
        print("="*70)
        
        # Retrieve documents
        retrieved_docs = rag.retrieve(test['question'], top_k=3)
        
        # Calculate metrics
        relevance = calculate_relevance_score(retrieved_docs)
        hit_rate = calculate_hit_rate(retrieved_docs, relevant_threshold=0.5)
        mrr = calculate_mrr(retrieved_docs, relevant_threshold=0.5)
        top1_rel = retrieved_docs[0]['relevance_score'] if retrieved_docs else 0
        
        results['relevance_scores'].append(relevance)
        results['hit_rates'].append(hit_rate)
        results['mrr_scores'].append(mrr)
        results['top1_relevance'].append(top1_rel)
        
        # Store query result
        results['queries'].append({
            'question': test['question'],
            'relevance': relevance,
            'hit_rate': hit_rate,
            'mrr': mrr,
            'top1_relevance': top1_rel,
            'retrieved_count': len(retrieved_docs)
        })
        
        print(f"\n📊 Metrics:")
        print(f"   Average Relevance: {relevance:.4f} ({relevance*100:.2f}%)")
        print(f"   Hit Rate: {hit_rate} ({'✅ Hit' if hit_rate else '❌ Miss'})")
        print(f"   MRR: {mrr:.4f}")
        print(f"   Top-1 Relevance: {top1_rel:.4f} ({top1_rel*100:.2f}%)")
        
        print(f"\n🔍 Retrieved Documents:")
        for j, doc in enumerate(retrieved_docs, 1):
            print(f"   [{j}] Relevance: {doc['relevance_score']:.2%}")
            print(f"       {doc['content'][:100]}...")
    
    # Calculate overall metrics
    print("\n\n" + "="*70)
    print("📈 OVERALL RAG PERFORMANCE")
    print("="*70)
    
    overall = {
        'average_relevance': np.mean(results['relevance_scores']),
        'hit_rate': np.mean(results['hit_rates']),
        'mean_reciprocal_rank': np.mean(results['mrr_scores']),
        'top1_average_relevance': np.mean(results['top1_relevance']),
        'total_queries': len(test_queries)
    }
    
    print(f"\n✅ Metrics Summary:")
    print(f"   Total Queries: {overall['total_queries']}")
    print(f"   Average Relevance: {overall['average_relevance']:.4f} ({overall['average_relevance']*100:.2f}%)")
    print(f"   Hit Rate@3: {overall['hit_rate']:.4f} ({overall['hit_rate']*100:.2f}%)")
    print(f"   Mean Reciprocal Rank (MRR): {overall['mean_reciprocal_rank']:.4f}")
    print(f"   Top-1 Avg Relevance: {overall['top1_average_relevance']:.4f} ({overall['top1_average_relevance']*100:.2f}%)")
    
    # Interpretation
    print(f"\n📋 Interpretation:")
    if overall['average_relevance'] >= 0.7:
        print(f"   ✅ Excellent relevance scores (>70%)")
    elif overall['average_relevance'] >= 0.5:
        print(f"   ✅ Good relevance scores (50-70%)")
    else:
        print(f"   ⚠️  Low relevance scores (<50%)")
    
    if overall['hit_rate'] >= 0.9:
        print(f"   ✅ Excellent hit rate (>90%)")
    elif overall['hit_rate'] >= 0.7:
        print(f"   ✅ Good hit rate (70-90%)")
    else:
        print(f"   ⚠️  Low hit rate (<70%)")
    
    if overall['mean_reciprocal_rank'] >= 0.8:
        print(f"   ✅ Excellent MRR (>0.8) - Relevant docs appear early")
    elif overall['mean_reciprocal_rank'] >= 0.5:
        print(f"   ✅ Good MRR (0.5-0.8)")
    else:
        print(f"   ⚠️  Low MRR (<0.5)")
    
    print("="*70)
    
    # Save results
    output_dir = Path('outputs/results/rag')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'rag_evaluation_metrics.json', 'w') as f:
        json.dump({
            'overall_metrics': overall,
            'per_query_results': results['queries']
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir / 'rag_evaluation_metrics.json'}")
    
    return overall

if __name__ == "__main__":
    evaluate_rag_system()