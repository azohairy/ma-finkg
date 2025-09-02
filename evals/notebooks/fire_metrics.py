import openai
import json
import os
from typing import List, Dict, Tuple, Any

def infer_type(entity: str, triplets: List[Tuple[str, str, str]], is_head: bool, ontology_path: str = None) -> str:
    if ontology_path is None:
        from ma_finkg.config import Config
        ontology_path = Config.get_ontology_file(ontology="fire")  # Fix: use fire ontology
    
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)["relations"]
    
    possible_types = set()
    for head, rel, tail in triplets:
        if (is_head and head == entity) or (not is_head and tail == entity):
            possible = ontology.get(rel, {}).get('head_types' if is_head else 'tail_types', [])
            if possible_types:
                possible_types &= set(possible)
            else:
                possible_types = set(possible)
    # Return first valid type if found, otherwise default to most common FIRE entity type
    return list(possible_types)[0] if possible_types else None

class SimpleFIREExtractor:
    """Simple GPT-3.5 few-shot baseline for FIRE relation extraction"""
    
    def __init__(self, model, api_key: str = None):
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        
    def extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        
        prompt = """
        Find the relation between the entities given in the context and produce a list of triplets containing two entities and their relations. Only find out the following relations 
        ActionBuy, Actionin, ActionSell, ActionMerge, Actionto, Constituentof , Designation, Employeeof, Locatedin, Productof, Propertyof, Quantity, Sector, Subsidiaryof, Value, ValueChangeDecreaseby, ValueChangeIncreaseby and Valuein 
        ActionMerge indicate two company or organizations enters into merger agreements to form a single entity.
        ActionBuy represents the action of purchasing/acquiring a Company, FinancialEntity, Product, or BusinessUnit by a Company or a Person. 
        Actionto represents the relation between the action entity and the entity on which the action has taken. 
        Constituentof relation denotes one financial entity is part of another financial entity. 
        Actionin indicates the Date associated with an Action entity, signifying the time of occurrence of the action. 
        ActionSell represents the action of selling a Company, FinancialEntity, Product, or BusinessUnit by a Company or a Person. 
        Employeeof denotes the past, present or future employment relationship between a Person and a Company. 
        Designation indicates the job title or position of a Person, or the Designation of a Company in the financial context, providing information about the role or responsibility of the entity. 
        Locatedin indicates the geographical location or country associated with an entity, specifying the place or region where the entity is located. Money and Quantity can be in the place where they were generated, lost, profited, etc. Note that a Company is only Located in a place if it based in that place. 
        Productof indicates a Product is manufactured, sold, offered, or marketed by a Company, establishing a relationship between the Company and the Product. 
        Propertyof serves as an umbrella relation” that indicates a general association between two entities, mainly representing ownership or part-of/composition relationships. This relation is used to connect two entities when a more specific relation is not yet defined. 
        Quantity represents the countable quantity a FinancialEntity, BusinessUnit or Product. 
        Sector indicates the economic sector or industry to which a Company belongs, providing information about the broad business area or category of the Company’s operations. 
        Subsidiaryof indicates that a Company is a subsidiary of a parent Company, either wholly or majority owned. Note that ”brands” are always considered subsidiaries of their parent Company. A highly occurring pattern is a parent company selling its subsidiary company, in which case the Subsidiaryof relation is not annotated. 
        Value represents a non-countable value of a FinancialEntity, BusinessUnit or Product such as a monetary value or a percentage. A Company can also have a Value relation, but only for monetary values such as indicating the net worth of a company or the sale price in an acquisition. 
        ValueChangeDecreaseby indicates the decrease in monetary value or quantity of a FinancialEntity. An additional more rare use-case is the Quantity of a BusinessUnit decreasing, such as number of employees or number of offices. 
        ValueChangeIncreaseby indicates the increase in value or quantity of a FinancialEntity. An additional more rare usecase is the Quantity of a BusinessUnit increasing, such as number of employees or number of offices. 
        Valuein indicates the Date associated with a Money or Quantity entity, providing information about the specific time period to which the Money or Quantity value is related. 
        Please find few examples below Context : Bank of America to Buy Merrill Lynch for $50 Billion Answer : 
        [['Bank of America', 'ActionBuy', 'Merrill Lynch'],
        ['Buy', 'Actionto', 'Merrill Lynch'],
        ['Merrill Lynch', 'Value', '$50 Billion']]

        Text: {text}

        Output (JSON format):
        [["head", "relation", "tail"], ...]
        
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt.format(text=text)}],
                temperature=0
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            if result_text.startswith('```json'):
                result_text = result_text[7:-3]
            elif result_text.startswith('```'):
                result_text = result_text[3:-3]
                
            relations = json.loads(result_text)
            return [tuple(rel) for rel in relations if len(rel) == 3]
            
        except Exception as e:
            print(f"Error in extraction: {e}")
            return []

def evaluate_simple_baseline(model_name, fire_test_processed: List[Dict], max_samples: int = 50, entity_text_only: bool = True) -> Dict:
    """Run evaluation on FIRE test set"""
    extractor = SimpleFIREExtractor(model_name)
    
    total_tp = total_fp = total_fn = 0
    total_ner_tp = total_ner_fp = total_ner_fn = 0
    total_relation_type_tp = total_relation_type_fp = total_relation_type_fn = 0
    processed = 0
    examples = []
    
    for sample in fire_test_processed[:max_samples]:
        # Get gold relations
        entities_by_index = {i: entity for i, entity in enumerate(sample['entities'])}
        gold_relations = set()
        
        for relation in sample['relations']:
            head_entity = entities_by_index.get(relation['head_id'])
            tail_entity = entities_by_index.get(relation['tail_id'])
            if head_entity and tail_entity:
                gold_relations.add((
                    head_entity['text'],
                    relation['type'],
                    tail_entity['text'] 
                ))
        
        # Get system predictions
        sys_relations = set()
        try:
            predictions = extractor.extract_relations(sample['sentence'])
            for head, rel, tail in predictions:
                # Verbatim match
                sys_relations.add((head, rel, tail))
        except:
            pass
        
        # Store example for display
        examples.append({
            'sentence': sample['sentence'],
            'gold': list(gold_relations),
            'predicted': list(sys_relations),
            'tp': len(gold_relations & sys_relations),
            'fp': len(sys_relations - gold_relations),
            'fn': len(gold_relations - sys_relations)
        })
        
        # Calculate triplet metrics (instance RE)
        tp = len(gold_relations & sys_relations)
        fp = len(sys_relations - gold_relations)
        fn = len(gold_relations - sys_relations)
        
        # Entity evaluation: baseline only extracts entity text from relations
        if entity_text_only:
            # Fair comparison: entity text only (baseline limitation)
            gold_ner = set(ent['text'] for ent in sample['entities'])
            sys_ner = set()
            # Extract entity text from relations
            for head, rel, tail in sys_relations:
                sys_ner.add(head)
                sys_ner.add(tail)
        else:
            gold_ner = set((ent['text'], ent['type']) for ent in sample['entities'])
            sys_ner = set()
            for head, rel, tail in sys_relations:
                sys_ner.add((head, "Company"))  
                sys_ner.add((tail, "Company"))   # Dummy type
        
        ner_tp = len(gold_ner & sys_ner)
        ner_fp = len(sys_ner - gold_ner)  
        ner_fn = len(gold_ner - sys_ner)
        
        # Calculate relation type metrics 
        gold_rels = {rel for head, rel, tail in gold_relations}
        sys_rels = {rel for head, rel, tail in sys_relations}
        
        relation_type_tp = len(gold_rels & sys_rels)
        relation_type_fp = len(sys_rels - gold_rels)
        relation_type_fn = len(gold_rels - sys_rels)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_ner_tp += ner_tp
        total_ner_fp += ner_fp
        total_ner_fn += ner_fn
        total_relation_type_tp += relation_type_tp
        total_relation_type_fp += relation_type_fp
        total_relation_type_fn += relation_type_fn
        processed += 1
        
        # Show details for first few samples
        if processed <= 3:
            print(f"\n=== Sample {processed} ===")
            print(f"Text: {sample['sentence']}")
            print(f"Gold: {list(gold_relations)}")
            print(f"Predicted: {list(sys_relations)}")
            print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        
        if processed % 10 == 0:
            print(f"Processed {processed}/{max_samples}")
    
    # Calculate final triplet metrics 
    re_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    re_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    re_f1 = 2 * re_precision * re_recall / (re_precision + re_recall) if re_precision + re_recall > 0 else 0
    
    # Calculate ner metrics
    ner_precision = total_ner_tp / (total_ner_tp + total_ner_fp) if total_ner_tp + total_ner_fp > 0 else 0
    ner_recall = total_ner_tp / (total_ner_tp + total_ner_fn) if total_ner_tp + total_ner_fn > 0 else 0
    ner_f1 = 2 * ner_precision * ner_recall / (ner_precision + ner_recall) if ner_precision + ner_recall > 0 else 0
    
    # Calculate relation type metrics  
    relation_type_precision = total_relation_type_tp / (total_relation_type_tp + total_relation_type_fp) if total_relation_type_tp + total_relation_type_fp > 0 else 0
    relation_type_recall = total_relation_type_tp / (total_relation_type_tp + total_relation_type_fn) if total_relation_type_tp + total_relation_type_fn > 0 else 0
    relation_type_f1 = 2 * relation_type_precision * relation_type_recall / (relation_type_precision + relation_type_recall) if relation_type_precision + relation_type_recall > 0 else 0
    
    # Export triplets comparison
    triplets_data = [{"sentence": ex['sentence'], "gold_triplets": ex['gold'], "predicted_triplets": ex['predicted']} for ex in examples]
    with open('fire_triplets_comparison.json', 'w') as f:
        json.dump(triplets_data, f, indent=2)
    
    return {
        're_precision': re_precision,
        're_recall': re_recall, 
        're_f1': re_f1,
        'ner_precision': ner_precision,
        'ner_recall': ner_recall,
        'ner_f1': ner_f1,
        'relation_type_precision': relation_type_precision,
        'relation_type_recall': relation_type_recall,
        'relation_type_f1': relation_type_f1,
        'total_re_tp': total_tp,
        'total_re_fp': total_fp,
        'total_re_fn': total_fn,
        'samples': processed,
        'examples': examples
    }

# Comprehensive FIRE Evaluation: NER, RE, and Triplets Metrics
# Multi-agent system Eval
def calculate_fire_metrics(fire_test_data, kg_system, max_samples=10, entity_text_only=False):
    
    # Normalization function for text matching
    def normalize_text(text):
        """Normalize text for more flexible matching"""
        return text.lower().strip().replace("  ", " ")
    
    # Counters for each metric type
    ner_tp = ner_fp = ner_fn = 0
    relation_type_tp = relation_type_fp = relation_type_fn = 0
    re_tp = re_fp = re_fn = 0
    
    processed_samples = 0
    comparison_data = [] 
    
    for i, sample in enumerate(fire_test_data[:max_samples]):
        if i % 10 == 0:
            print(f"Processing sample {i+1}/{min(max_samples, len(fire_test_data))}...")
        
        text = sample['sentence'].strip()
        
        # Get system predictions
        result = kg_system.construct_kg(text)
        sys_entities = result['finalize'].get("revised_entities", []) if 'finalize' in result else []
        sys_triples = result['finalize'].get("revised_triples", []) if 'finalize' in result else []
        
        # Extract gold standard data
        gold_entities = sample['entities']
        entities_by_idx = {j: entity for j, entity in enumerate(gold_entities)}
        
        gold_triples = []
        for relation in sample['relations']:
            head_entity = entities_by_idx.get(relation['head_id'])
            tail_entity = entities_by_idx.get(relation['tail_id'])
            if head_entity and tail_entity:
                gold_triples.append((
                    head_entity['text'],
                    relation['type'],
                    tail_entity['text']
                ))
        
        # Convert sys triples to comparable format for export
        sys_triples_list = []
        for triple in sys_triples:
            if hasattr(triple, 'head') and hasattr(triple, 'relation') and hasattr(triple, 'tail'):
                sys_triples_list.append([
                    triple.head,
                    triple.relation,
                    triple.tail
                ])
        
        # Convert gold triples to list format for consistency
        gold_triples_list = [[head, rel, tail] for head, rel, tail in gold_triples]
        
        # Store comparison data for this sample
        comparison_data.append({
            "sentence": text,
            "gold_triples": gold_triples_list,
            "sys_triples": sys_triples_list
        })
        
        # 1. NER EVALUATION: Entity-level matching
        if entity_text_only:
            gold_ner = set(normalize_text(ent['text']) for ent in gold_entities)
            sys_ner = set()
            for entity in sys_entities:
                if hasattr(entity, 'text'):
                    sys_ner.add(normalize_text(entity.text))
        else:
            # Full NER: entity text + type (normalized)
            gold_ner = set((normalize_text(ent['text']), ent['type']) for ent in gold_entities)
            sys_ner = set()
            for entity in sys_entities:
                if hasattr(entity, 'text') and hasattr(entity, 'entity_type'):
                    sys_ner.add((normalize_text(entity.text), entity.entity_type))
        
        ner_tp += len(gold_ner & sys_ner)
        ner_fp += len(sys_ner - gold_ner)
        ner_fn += len(gold_ner - sys_ner)
        
        # 2. RELATION TYPE EVALUATION: Unique relation types
        gold_relations = set(rel[1] for rel in gold_triples)
        sys_relations = set()
        
        for triple in sys_triples:
            if hasattr(triple, 'relation'):
                sys_relations.add(triple.relation)
        
        relation_type_tp += len(gold_relations & sys_relations)
        relation_type_fp += len(sys_relations - gold_relations)
        relation_type_fn += len(gold_relations - sys_relations)
        
        # 3. RE EVALUATION: Complete triplet matching with normalization
        # Normalize gold triples
        gold_triplet_set = set()
        for head, rel, tail in gold_triples:
            gold_triplet_set.add((
                normalize_text(head),
                rel,  # Keep relation as-is for now
                normalize_text(tail)
            ))
        
        # Normalize system triples  
        sys_triplet_set = set()
        for triple in sys_triples:
            if hasattr(triple, 'head') and hasattr(triple, 'relation') and hasattr(triple, 'tail'):
                sys_triplet_set.add((
                    normalize_text(triple.head),
                    triple.relation,  # Keep relation as-is for now
                    normalize_text(triple.tail)
                ))
        
        sample_tp = len(gold_triplet_set & sys_triplet_set)
        sample_fp = len(sys_triplet_set - gold_triplet_set) 
        sample_fn = len(gold_triplet_set - sys_triplet_set)
        
        # DEBUG: Print detailed mismatches for first few samples
        if i < 3:  # Show details for first 3 samples
            print(f"\n=== DEBUG SAMPLE {i+1} ===" + (" [TEXT-ONLY]" if entity_text_only else " [TEXT+TYPE]"))
            print(f"NER: Gold entities ({len(gold_ner)}): {list(gold_ner)})")
            print(f"NER: Sys entities ({len(sys_ner)}): {list(sys_ner)})")
            print(f"Gold triples ({len(gold_triplet_set)}): {list(gold_triplet_set)[:3]}")
            print(f"Sys triples ({len(sys_triplet_set)}): {list(sys_triplet_set)[:3]}")
            if sample_tp > 0:
                matches = gold_triplet_set & sys_triplet_set
                print(f"MATCHES ({len(matches)}): {list(matches)}")
            if sample_fp > 0:
                fps = sys_triplet_set - gold_triplet_set
                print(f"FALSE POSITIVES ({len(fps)}): {list(fps)[:3]}")
            if sample_fn > 0:
                fns = gold_triplet_set - sys_triplet_set
                print(f"MISSED ({len(fns)}): {list(fns)[:3]}")
        
        re_tp += sample_tp
        re_fp += sample_fp
        re_fn += sample_fn
        
        processed_samples += 1
    
    # Calculate metrics for each type
    def compute_metrics(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1
    
    ner_p, ner_r, ner_f1 = compute_metrics(ner_tp, ner_fp, ner_fn)
    relation_type_p, relation_type_r, relation_type_f1 = compute_metrics(relation_type_tp, relation_type_fp, relation_type_fn)
    re_p, re_r, re_f1 = compute_metrics(re_tp, re_fp, re_fn)
    
    
    return {
        'processed_samples': processed_samples,
        'ner': {
            'precision': ner_p,
            'recall': ner_r, 
            'f1': ner_f1,
            'tp': ner_tp,
            'fp': ner_fp,
            'fn': ner_fn
        },
        'relation_type': {
            'precision': relation_type_p,
            'recall': relation_type_r,
            'f1': relation_type_f1,
            'tp': relation_type_tp,
            'fp': relation_type_fp,
            'fn': relation_type_fn
        },
        're': {
            'precision': re_p,
            'recall': re_r,
            'f1': re_f1,
            'tp': re_tp,
            'fp': re_fp,
            'fn': re_fn
        },
        'comparison_data': {
            'data': comparison_data
        }
    }