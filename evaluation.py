#!/usr/bin/env python3
"""
Evaluation system for RAG Q&A performance comparison.
Tests baseline vs hybrid reranker with comprehensive metrics.
"""

import sys
import os
import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass

# Add src to path
sys.path.append('src')

from api import QAService

@dataclass
class TestQuestion:
    """Represents a test question with expected characteristics."""
    question: str
    category: str
    difficulty: str  # easy, medium, hard
    expected_sources: List[str]  # Expected source types/keywords
    description: str

@dataclass
class EvaluationResult:
    """Results of evaluating a single question."""
    question: str
    baseline_response: Dict[str, Any]
    hybrid_response: Dict[str, Any]
    baseline_time: float
    hybrid_time: float
    metrics: Dict[str, float]

class RAGEvaluator:
    """Comprehensive evaluation system for RAG Q&A."""
    
    def __init__(self, qa_service: QAService):
        """Initialize evaluator with QA service."""
        self.qa_service = qa_service
        self.test_questions = self._create_test_questions()
    
    def _create_test_questions(self) -> List[TestQuestion]:
        """Create comprehensive test questions covering different scenarios."""
        return [
            TestQuestion(
                question="What are the basic safety requirements for industrial machinery?",
                category="basic_safety",
                difficulty="easy",
                expected_sources=["machinery", "safety", "requirements"],
                description="General safety principles - should find broad regulatory content"
            ),
            TestQuestion(
                question="How do you perform a risk assessment according to EN ISO 12100?",
                category="risk_assessment", 
                difficulty="medium",
                expected_sources=["ISO 12100", "risk assessment", "standard"],
                description="Specific standard reference - tests precise document retrieval"
            ),
            TestQuestion(
                question="What are the requirements for emergency stop systems in industrial machinery?",
                category="emergency_systems",
                difficulty="medium", 
                expected_sources=["emergency stop", "EN ISO 13850", "safety"],
                description="Specific safety system - should find technical specifications"
            ),
            TestQuestion(
                question="Explain the difference between SIL and PL safety ratings",
                category="safety_ratings",
                difficulty="hard",
                expected_sources=["SIL", "PL", "safety integrity", "performance level"],
                description="Technical comparison - tests nuanced understanding"
            ),
            TestQuestion(
                question="What machine guarding methods are required for point of operation protection?",
                category="machine_guarding",
                difficulty="medium",
                expected_sources=["machine guarding", "point of operation", "protection"],
                description="Specific protective measures - should find OSHA and equipment standards"
            ),
            TestQuestion(
                question="How should lockout/tagout procedures be implemented for machinery maintenance?",
                category="loto_procedures", 
                difficulty="medium",
                expected_sources=["lockout", "tagout", "LOTO", "maintenance"],
                description="Procedural safety - tests operational guidance retrieval"
            ),
            TestQuestion(
                question="What are the Category 4 requirements according to EN ISO 13849-1?",
                category="functional_safety",
                difficulty="hard",
                expected_sources=["Category 4", "EN ISO 13849", "functional safety"],
                description="Specific technical category - tests detailed standard knowledge"
            ),
            TestQuestion(
                question="How do you calculate the required Performance Level (PLr) for a safety function?",
                category="safety_calculations",
                difficulty="hard", 
                expected_sources=["Performance Level", "PLr", "calculation", "safety function"],
                description="Technical calculation - tests detailed procedural knowledge"
            )
        ]
    
    def _evaluate_response_quality(self, question: TestQuestion, response: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the quality of a response with multiple metrics."""
        metrics = {}
        
        # Basic response metrics
        metrics['has_answer'] = 1.0 if response.get('answer') else 0.0
        metrics['confidence'] = response.get('confidence', 0.0)
        metrics['num_contexts'] = len(response.get('contexts', []))
        
        # Content relevance (simplified heuristic)
        answer = response.get('answer', '').lower()
        contexts = response.get('contexts', [])
        
        # Check if expected source keywords appear
        expected_matches = 0
        for expected in question.expected_sources:
            if any(expected.lower() in (answer + ' '.join([c.get('text', '') for c in contexts])).lower() 
                   for expected in [expected]):
                expected_matches += 1
        
        metrics['source_relevance'] = expected_matches / len(question.expected_sources) if question.expected_sources else 0.0
        
        # Citation quality
        metrics['citation_count'] = len([c for c in contexts if c.get('url')])
        metrics['source_diversity'] = len(set(c.get('source', '') for c in contexts))
        
        # Answer length appropriateness
        answer_length = len(answer) if answer else 0
        if 50 <= answer_length <= 300:
            metrics['length_score'] = 1.0
        elif answer_length > 0:
            metrics['length_score'] = 0.5
        else:
            metrics['length_score'] = 0.0
        
        return metrics
    
    def _compare_responses(self, baseline: Dict[str, Any], hybrid: Dict[str, Any]) -> Dict[str, Any]:
        """Compare baseline vs hybrid responses."""
        comparison = {}
        
        # Confidence comparison
        baseline_conf = baseline.get('confidence', 0.0)
        hybrid_conf = hybrid.get('confidence', 0.0)
        comparison['confidence_improvement'] = hybrid_conf - baseline_conf
        
        # Answer quality comparison
        baseline_has_answer = bool(baseline.get('answer'))
        hybrid_has_answer = bool(hybrid.get('answer'))
        
        if baseline_has_answer and hybrid_has_answer:
            comparison['both_answered'] = True
            comparison['answer_change'] = baseline.get('answer') != hybrid.get('answer')
        elif hybrid_has_answer and not baseline_has_answer:
            comparison['hybrid_improvement'] = True
        elif baseline_has_answer and not hybrid_has_answer:
            comparison['hybrid_degradation'] = True
        else:
            comparison['both_failed'] = True
        
        # Ranking changes
        baseline_sources = [c.get('source', '') for c in baseline.get('contexts', [])]
        hybrid_sources = [c.get('source', '') for c in hybrid.get('contexts', [])]
        comparison['ranking_change'] = baseline_sources != hybrid_sources
        
        return comparison
    
    def evaluate_question(self, question: TestQuestion, k: int = 5) -> EvaluationResult:
        """Evaluate a single question with both methods."""
        print(f"  Evaluating: {question.question[:60]}...")
        
        # Test baseline
        start_time = time.time()
        baseline_response = self.qa_service.ask(question.question, k=k, mode='baseline')
        baseline_time = time.time() - start_time
        
        # Test hybrid  
        start_time = time.time()
        hybrid_response = self.qa_service.ask(question.question, k=k, mode='hybrid')
        hybrid_time = time.time() - start_time
        
        # Evaluate responses
        baseline_metrics = self._evaluate_response_quality(question, baseline_response)
        hybrid_metrics = self._evaluate_response_quality(question, hybrid_response)
        comparison = self._compare_responses(baseline_response, hybrid_response)
        
        # Combined metrics
        metrics = {
            'baseline': baseline_metrics,
            'hybrid': hybrid_metrics,
            'comparison': comparison,
            'performance': {
                'baseline_time': baseline_time,
                'hybrid_time': hybrid_time
            }
        }
        
        return EvaluationResult(
            question=question.question,
            baseline_response=baseline_response,
            hybrid_response=hybrid_response,
            baseline_time=baseline_time,
            hybrid_time=hybrid_time,
            metrics=metrics
        )
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation on all test questions."""
        print("🧪 Starting Comprehensive RAG Evaluation")
        print("=" * 60)
        
        results = []
        category_stats = {}
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n📝 Test {i}/{len(self.test_questions)} - {question.category.upper()}")
            print(f"   Difficulty: {question.difficulty}")
            print(f"   Description: {question.description}")
            
            result = self.evaluate_question(question)
            results.append(result)
            
            # Track category performance
            if question.category not in category_stats:
                category_stats[question.category] = []
            category_stats[question.category].append(result)
        
        # Calculate overall statistics
        overall_stats = self._calculate_overall_stats(results)
        
        evaluation_report = {
            'test_summary': {
                'total_questions': len(self.test_questions),
                'categories': list(category_stats.keys()),
                'difficulty_distribution': {
                    d: len([q for q in self.test_questions if q.difficulty == d]) 
                    for d in ['easy', 'medium', 'hard']
                }
            },
            'overall_performance': overall_stats,
            'category_breakdown': {
                cat: self._calculate_category_stats(cat_results) 
                for cat, cat_results in category_stats.items()
            },
            'detailed_results': [
                {
                    'question': r.question,
                    'category': next(q.category for q in self.test_questions if q.question == r.question),
                    'difficulty': next(q.difficulty for q in self.test_questions if q.question == r.question),
                    'baseline_confidence': r.baseline_response.get('confidence', 0.0),
                    'hybrid_confidence': r.hybrid_response.get('confidence', 0.0),
                    'confidence_improvement': r.metrics['comparison'].get('confidence_improvement', 0.0),
                    'baseline_answered': bool(r.baseline_response.get('answer')),
                    'hybrid_answered': bool(r.hybrid_response.get('answer')),
                    'ranking_changed': r.metrics['comparison'].get('ranking_change', False),
                    'response_times': {
                        'baseline': r.baseline_time,
                        'hybrid': r.hybrid_time
                    }
                }
                for r in results
            ]
        }
        
        return evaluation_report
    
    def _calculate_overall_stats(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate overall performance statistics."""
        if not results:
            return {}
        
        baseline_answered = sum(1 for r in results if r.baseline_response.get('answer'))
        hybrid_answered = sum(1 for r in results if r.hybrid_response.get('answer'))
        
        baseline_avg_conf = sum(r.baseline_response.get('confidence', 0.0) for r in results) / len(results)
        hybrid_avg_conf = sum(r.hybrid_response.get('confidence', 0.0) for r in results) / len(results)
        
        confidence_improvements = [
            r.metrics['comparison'].get('confidence_improvement', 0.0) for r in results
        ]
        
        ranking_changes = sum(
            1 for r in results if r.metrics['comparison'].get('ranking_change', False)
        )
        
        return {
            'answer_rates': {
                'baseline': baseline_answered / len(results),
                'hybrid': hybrid_answered / len(results),
                'improvement': (hybrid_answered - baseline_answered) / len(results)
            },
            'average_confidence': {
                'baseline': baseline_avg_conf,
                'hybrid': hybrid_avg_conf,
                'improvement': hybrid_avg_conf - baseline_avg_conf
            },
            'confidence_improvements': {
                'positive': sum(1 for imp in confidence_improvements if imp > 0),
                'negative': sum(1 for imp in confidence_improvements if imp < 0),
                'unchanged': sum(1 for imp in confidence_improvements if imp == 0),
                'average_improvement': sum(confidence_improvements) / len(confidence_improvements)
            },
            'ranking_changes': {
                'total': ranking_changes,
                'percentage': ranking_changes / len(results)
            },
            'response_times': {
                'baseline_avg': sum(r.baseline_time for r in results) / len(results),
                'hybrid_avg': sum(r.hybrid_time for r in results) / len(results)
            }
        }
    
    def _calculate_category_stats(self, category_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate statistics for a specific category."""
        if not category_results:
            return {}
        
        return {
            'question_count': len(category_results),
            'answer_rate': {
                'baseline': sum(1 for r in category_results if r.baseline_response.get('answer')) / len(category_results),
                'hybrid': sum(1 for r in category_results if r.hybrid_response.get('answer')) / len(category_results)
            },
            'avg_confidence': {
                'baseline': sum(r.baseline_response.get('confidence', 0.0) for r in category_results) / len(category_results),
                'hybrid': sum(r.hybrid_response.get('confidence', 0.0) for r in category_results) / len(category_results)
            }
        }
    
    def print_evaluation_summary(self, report: Dict[str, Any]) -> None:
        """Print a formatted evaluation summary."""
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        # Overall performance
        overall = report['overall_performance']
        print(f"\n🎯 Overall Performance:")
        print(f"  Questions answered:")
        print(f"    Baseline: {overall['answer_rates']['baseline']:.1%}")
        print(f"    Hybrid:   {overall['answer_rates']['hybrid']:.1%}")
        print(f"    Change:   {overall['answer_rates']['improvement']:+.1%}")
        
        print(f"\n  Average confidence:")
        print(f"    Baseline: {overall['average_confidence']['baseline']:.3f}")
        print(f"    Hybrid:   {overall['average_confidence']['hybrid']:.3f}")
        print(f"    Change:   {overall['average_confidence']['improvement']:+.3f}")
        
        conf_imp = overall['confidence_improvements']
        print(f"\n  Confidence changes:")
        print(f"    Improved:  {conf_imp['positive']} questions")
        print(f"    Degraded:  {conf_imp['negative']} questions")
        print(f"    Unchanged: {conf_imp['unchanged']} questions")
        
        print(f"\n  Ranking changes: {overall['ranking_changes']['percentage']:.1%} of questions")
        
        # Category breakdown
        print(f"\n📂 Performance by Category:")
        for category, stats in report['category_breakdown'].items():
            print(f"\n  {category.replace('_', ' ').title()}:")
            print(f"    Answer rate: {stats['answer_rate']['baseline']:.1%} → {stats['answer_rate']['hybrid']:.1%}")
            print(f"    Confidence:  {stats['avg_confidence']['baseline']:.3f} → {stats['avg_confidence']['hybrid']:.3f}")
        
        # Performance insights
        print(f"\n🔍 Key Findings:")
        
        if overall['answer_rates']['improvement'] > 0:
            print(f"  ✅ Hybrid reranker improved answer rate by {overall['answer_rates']['improvement']:.1%}")
        elif overall['answer_rates']['improvement'] < 0:
            print(f"  ⚠️ Hybrid reranker decreased answer rate by {abs(overall['answer_rates']['improvement']):.1%}")
        else:
            print(f"  ➡️ No change in overall answer rate")
        
        if overall['average_confidence']['improvement'] > 0.05:
            print(f"  ✅ Significant confidence improvement (+{overall['average_confidence']['improvement']:.3f})")
        elif overall['average_confidence']['improvement'] < -0.05:
            print(f"  ⚠️ Confidence decreased (-{abs(overall['average_confidence']['improvement']):.3f})")
        else:
            print(f"  ➡️ Minimal confidence change ({overall['average_confidence']['improvement']:+.3f})")
        
        if overall['ranking_changes']['percentage'] > 0.5:
            print(f"  🔄 High reranking activity ({overall['ranking_changes']['percentage']:.1%} of questions)")
        else:
            print(f"  🔄 Moderate reranking activity ({overall['ranking_changes']['percentage']:.1%} of questions)")

def main():
    """Run the evaluation."""
    print("🚀 Initializing RAG Q&A System for Evaluation...")
    
    # Initialize QA service
    qa_service = QAService(db_path="data/rag_database.db")
    
    # Run evaluation
    evaluator = RAGEvaluator(qa_service)
    report = evaluator.run_full_evaluation()
    
    # Print summary
    evaluator.print_evaluation_summary(report)
    
    # Save detailed report
    output_file = "evaluation_report.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed report saved to: {output_file}")
    print("\n✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main()
