import json
from typing import List, Dict, Tuple, Any

def normalize_refind_relation(relation: str) -> str:
    if ':' in relation:
        return relation.split(':')[-1]
    return relation

def calculate_macro_f1(predictions_by_relation: dict, exclude_no_relation: bool = True) -> dict:
    per_relation_f1 = {}
    valid_f1_scores = []
    
    for relation_type, counts in predictions_by_relation.items():
        if exclude_no_relation and relation_type == "no_relation":
            continue
            
        tp = counts['tp']
        fp = counts['fp'] 
        fn = counts['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_relation_f1[relation_type] = {
            'precision': precision,
            'recall': recall, 
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        
        # Include relations that appear in gold (tp + fn > 0) or predictions (tp + fp > 0)
        if tp + fn > 0 or tp + fp > 0:  # Relation appears in gold or system predictions
            valid_f1_scores.append(f1)
    
    macro_f1 = sum(valid_f1_scores) / len(valid_f1_scores) if valid_f1_scores else 0.0
    
    return {
        'macro_f1': macro_f1,
        'per_relation': per_relation_f1,
        'num_relations': len(valid_f1_scores)
    }

def evaluate_track_b_direct_classification(refind_test_data, kg_system, max_samples=10):
    """
    Track B: Direct entity pair classification (Li et al. paper methodology).

    """
    predictions_by_relation = {}  # For macro F1 calculation
    processed_samples = 0
    comparison_data = []
    
    for i, sample in enumerate(refind_test_data[:max_samples]):
        if i % 5 == 0:
            print(f"Track B: Processing sample {i+1}/{min(max_samples, len(refind_test_data))}...")
        
        sentence = sample['sentence'].strip()
        gold_relation = normalize_refind_relation(sample['gold_relation'])
        
        # Extract entity pair information
        e1 = sample['entities'][0]  # First entity (head)
        e2 = sample['entities'][1]  # Second entity (tail)
        
        # Direct classification using new classify_relation method
        try:
            predicted_relation = kg_system.classify_relation(
                sentence, e1['text'], e1['type'], e2['text'], e2['type']
            )
            predicted_relation = normalize_refind_relation(predicted_relation)
            
            # DEBUG: Show first few classifications
            if i < 3:
                print(f"Track B Sample {i+1}: '{e1['text']}' → '{e2['text']}' = {predicted_relation} (gold: {gold_relation})")
                
        except Exception as e:
            print(f"Error in Track B classification: {e}")
            predicted_relation = "no_relation"
        
        # Track per-relation metrics for Macro-F1
        # Initialize relation counters if needed
        for relation in [gold_relation, predicted_relation]:
            if relation not in predictions_by_relation:
                predictions_by_relation[relation] = {'tp': 0, 'fp': 0, 'fn': 0}
        
        # Update counts based on prediction
        if gold_relation == predicted_relation:
            predictions_by_relation[gold_relation]['tp'] += 1
        else:
            predictions_by_relation[gold_relation]['fn'] += 1
            predictions_by_relation[predicted_relation]['fp'] += 1
        
        # Store detailed comparison
        comparison_data.append({
            "sentence": sentence,
            "e1": f"{e1['text']} ({e1['type']})",
            "e2": f"{e2['text']} ({e2['type']})", 
            "gold_relation": gold_relation,
            "predicted_relation": predicted_relation,
            "correct": gold_relation == predicted_relation
        })
        
        
        processed_samples += 1
    
    # Calculate Macro-F1 metrics
    macro_results = calculate_macro_f1(predictions_by_relation)
    
    return {
        'processed_samples': processed_samples,
        'macro_f1': macro_results['macro_f1'],
        'per_relation_f1': macro_results['per_relation'],
        'num_relations': macro_results['num_relations'],
        'predictions_by_relation': predictions_by_relation,
        'comparison_data': comparison_data
    }

def evaluate_track_a_full_pipeline(refind_test_data, kg_system, max_samples=10):
    """
    Track A: Full pipeline evaluation with proper span matching (System Capability).
    """
    
    def normalize_text(text):
        """Normalize text for more flexible matching"""
        return text.lower().strip().replace("  ", " ")
    
    # Counters for strict F1 relation extraction
    strict_tp = strict_fp = strict_fn = 0  # Excludes true negatives
    true_negatives = 0  # Count excluded samples for reporting
    
    # Standard NER counters (unaffected by strict F1)
    ner_tp = ner_fp = ner_fn = 0
    
    processed_samples = 0
    comparison_data = [] 
    
    for i, sample in enumerate(refind_test_data[:max_samples]):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{min(max_samples, len(refind_test_data))}...")
        
        text = sample['sentence'].strip()
        gold_relation = sample['gold_relation']  # Original relation including no_relation
        
        # Get system predictions via full pipeline
        result = kg_system.construct_kg(text)
        sys_entities = result['finalize'].get("revised_entities", []) if 'finalize' in result else []
        sys_triples = result['finalize'].get("revised_triples", []) if 'finalize' in result else []
        
        # Extract gold entity pair information  
        gold_entities = sample['entities']
        gold_e1_text = normalize_text(gold_entities[0]['text'])
        gold_e2_text = normalize_text(gold_entities[1]['text'])
        
        # Find predicted relation that matches the gold (e1,e2) pair exactly
        predicted_relation = "no_relation"  # Default
        for triple in sys_triples:
            if hasattr(triple, 'head') and hasattr(triple, 'tail') and hasattr(triple, 'relation'):
                sys_head = normalize_text(triple.head)
                sys_tail = normalize_text(triple.tail)
                
                # Check if this triple matches the gold entity pair (directional)
                if sys_head == gold_e1_text and sys_tail == gold_e2_text:
                    predicted_relation = normalize_refind_relation(triple.relation)
                    break  # Found matching triple for gold pair
        
        # Normalize gold relation
        normalized_gold = normalize_refind_relation(gold_relation)
        
        # NER EVALUATION: Compare system entities to gold entities (all samples)
        gold_ner = set((normalize_text(ent['text']), ent['type']) for ent in sample['entities'])
        sys_ner = set()
        
        for entity in sys_entities:
            if hasattr(entity, 'text') and hasattr(entity, 'entity_type'):
                sys_ner.add((normalize_text(entity.text), entity.entity_type))
        
        sample_ner_tp = len(gold_ner & sys_ner)
        sample_ner_fp = len(sys_ner - gold_ner)
        sample_ner_fn = len(gold_ner - sys_ner)
        
        ner_tp += sample_ner_tp
        ner_fp += sample_ner_fp
        ner_fn += sample_ner_fn
        
        # DEBUG: Show first few samples for Track A
        if i < 3:
            print(f"Track A Sample {i+1}: Gold='{gold_e1_text}' → '{gold_e2_text}' ({normalized_gold})")
            print(f"  System triples: {len(sys_triples)}")
            for j, triple in enumerate(sys_triples[:3]):  # Show first 3 triples
                if hasattr(triple, 'head') and hasattr(triple, 'tail') and hasattr(triple, 'relation'):
                    print(f"    {j+1}: '{normalize_text(triple.head)}' → '{normalize_text(triple.tail)}' ({triple.relation})")
            print(f"  Predicted: {predicted_relation}")
            print(f"  NER: Gold={len(gold_ner)}, Sys={len(sys_ner)}, TP={sample_ner_tp}")
            print()
        
        # STRICT F1 CATEGORIZATION (excludes true negatives)
        if normalized_gold == "no_relation" and predicted_relation == "no_relation":
            # TRUE NEGATIVE: Both gold and predicted are no_relation → EXCLUDE from F1
            true_negatives += 1
        elif normalized_gold != "no_relation" and predicted_relation != "no_relation" and normalized_gold == predicted_relation:
            # TRUE POSITIVE: Both are real relations and match
            strict_tp += 1
        elif normalized_gold == "no_relation" and predicted_relation != "no_relation":
            # FALSE POSITIVE: Gold is no_relation but system predicted a relation
            strict_fp += 1
        elif normalized_gold != "no_relation" and predicted_relation == "no_relation":
            # FALSE NEGATIVE: Gold is a real relation but system predicted no_relation
            strict_fn += 1
        elif normalized_gold != "no_relation" and predicted_relation != "no_relation" and normalized_gold != predicted_relation:
            # FALSE POSITIVE + FALSE NEGATIVE: Both are real relations but different
            strict_fp += 1
            strict_fn += 1
        
        # Store detailed comparison data
        sys_triples_list = []
        for triple in sys_triples:
            if hasattr(triple, 'head') and hasattr(triple, 'relation') and hasattr(triple, 'tail'):
                sys_triples_list.append([
                    triple.head,
                    normalize_refind_relation(triple.relation),
                    triple.tail
                ])
        
        comparison_data.append({
            "sentence": text,
            "gold_relation": normalized_gold,
            "predicted_relation": predicted_relation,
            "strict_category": "TN" if normalized_gold == "no_relation" and predicted_relation == "no_relation" else (
                "TP" if normalized_gold != "no_relation" and predicted_relation != "no_relation" and normalized_gold == predicted_relation else (
                    "FP" if normalized_gold == "no_relation" and predicted_relation != "no_relation" else "FN"
                )
            ),
            "sys_triples": sys_triples_list
        })
        
        # DEBUG: Print detailed analysis for first few samples
        if i < 5:
            print(f"\n=== DEBUG SAMPLE {i+1} ===")
            print(f"Text: {text[:100]}...")
            print(f"Gold relation: {normalized_gold}")
            print(f"Predicted relation: {predicted_relation}")
            category = comparison_data[-1]["strict_category"] 
            print(f"Strict F1 category: {category}")
            if category == "TN":
                print("  → EXCLUDED from strict F1 calculation")
            else:
                print("  → INCLUDED in strict F1 calculation")
        
        processed_samples += 1
    
    # Calculate strict F1 metrics (excluding true negatives)
    def compute_strict_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
    
    # Standard metrics for NER
    def compute_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
    
    ner_p, ner_r, ner_f1 = compute_metrics(ner_tp, ner_fp, ner_fn)
    strict_p, strict_r, strict_f1 = compute_strict_metrics(strict_tp, strict_fp, strict_fn)
    
    return {
        'processed_samples': processed_samples,
        'true_negatives_excluded': true_negatives,
        'ner': {
            'precision': ner_p,
            'recall': ner_r, 
            'f1': ner_f1,
            'tp': ner_tp,
            'fp': ner_fp,
            'fn': ner_fn
        },
        'strict_f1_relation': {
            'precision': strict_p,
            'recall': strict_r,
            'f1': strict_f1,
            'tp': strict_tp,
            'fp': strict_fp,
            'fn': strict_fn,
            'excluded_tn': true_negatives
        },
        'comparison_data': {
            'data': comparison_data
        }
    }

