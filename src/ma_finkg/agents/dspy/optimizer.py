"""
DSPy Auto-Optimization Module

Surgical implementation of DSPy's auto-optimization capabilities.
Keeps optimization logic separate from core extraction agents.
"""

import dspy
import dspy.teleprompt
from typing import List, Dict, Any


def f1_metric(gold, pred, trace=None):
    """F1 metric with debugging to understand MIPROv2's 93% vs reality disconnect."""
    if not hasattr(pred, 'tail_entities') or not pred.tail_entities:
        score = 0.8 if (hasattr(gold, 'tail_entities') and not gold.tail_entities) else 0.0
        print(f"[METRIC] Empty pred, gold_empty={not hasattr(gold, 'tail_entities') or not gold.tail_entities}, score={score}")
        return score
        
    predicted_tails = set(t.lower().strip() for t in pred.tail_entities)
    gold_tails = set(t.lower().strip() for t in gold.tail_entities) if hasattr(gold, 'tail_entities') else set()
    
    if not gold_tails and not predicted_tails:
        print(f"[METRIC] Both empty, score=1.0")
        return 1.0
    if not gold_tails or not predicted_tails:
        print(f"[METRIC] One empty - gold:{gold_tails}, pred:{predicted_tails}, score=0.0")
        return 0.0
        
    tp = len(predicted_tails & gold_tails)
    fp = len(predicted_tails - gold_tails) 
    fn = len(gold_tails - predicted_tails)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    print(f"[METRIC] TP={tp}, FP={fp}, FN={fn}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
    print(f"[METRIC] Gold: {gold_tails}, Pred: {predicted_tails}")
    
    return f1


class DSPyOptimizer:
    """Surgical DSPy optimizer for NYT11-HRL relation extraction."""
    
    def __init__(self, max_demos=2, max_examples=20, validation_examples=15):
        self.max_demos = max_demos
        self.max_examples = max_examples
        self.validation_examples = validation_examples
        
    def optimize_predictor(self, predictor, train_examples, val_examples=None):
        """Optimize a DSPy predictor using train/val split.""" 
        if not train_examples:
            return predictor
            
        try:
            print(f"[DSPy OPT] Optimizing with {len(train_examples)} train + {len(val_examples or [])} val examples")
            
            # Use MIPROv2 for instruction optimization
            optimizer = dspy.teleprompt.MIPROv2(metric=f1_metric)
            optimized_predictor = optimizer.compile(
                predictor, 
                trainset=train_examples,
                valset=val_examples,
                requires_permission_to_run=False
            )
            print("[DSPy OPT] Optimization completed")
            return optimized_predictor
            
        except Exception as e:
            print(f"[DSPy OPT] Optimization failed: {e}")
            return predictor
    
    def create_dspy_example(self, text, head_entity, relation_type, tail_entities):
        """Create DSPy training example from evaluation data."""
        return dspy.Example(
            text=text,
            head_entity=head_entity, 
            relation_type=relation_type,
            tail_entities=tail_entities
        ).with_inputs('text', 'head_entity', 'relation_type')