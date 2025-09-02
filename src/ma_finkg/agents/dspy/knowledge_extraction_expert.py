"""
DSPy-based Knowledge Extraction Expert - Orchestrates NER and RE with structured outputs.

This agent provides the same interface as the YAML-based version but uses DSPy
for structured prompt optimization and reliable JSON outputs.
"""

import dspy
from typing import Dict, Any, List, Tuple, Set
import signal
import functools

from ma_finkg.models import GraphState, Entity, Triple, FinancialOntology
from ma_finkg.utils import print_progress, get_elapsed_time, spinner
from ma_finkg.agents.dspy.optimizer import DSPyOptimizer


def timeout_handler(signum, frame):
    raise TimeoutError("DSPy call timed out")

def with_timeout(timeout_seconds=30):
    """Decorator to add timeout protection to function calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set timeout alarm
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Reset alarm
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


class EntityExtractionSignature(dspy.Signature):
    """Extract entities of a specific type from text."""
    text: str = dspy.InputField(desc="Text to extract entities from")
    entity_type: str = dspy.InputField(desc="Type of entities to extract")
    entities: List[str] = dspy.OutputField(desc="List of extracted entity names")


class HeadExtractionSignature(dspy.Signature):
    """Extract head entities for relation extraction."""
    text: str = dspy.InputField(desc="Text to analyze")
    relation_type: str = dspy.InputField(desc="Target relation type")
    head_entities: List[str] = dspy.OutputField(desc="List of potential head entities")


class TailExtractionSignature(dspy.Signature):
    """Extract tail entities for a given head entity and relation type."""
    text: str = dspy.InputField(desc="Text to analyze")
    head_entity: str = dspy.InputField(desc="Head entity")
    relation_type: str = dspy.InputField(desc="Target relation type")
    tail_entities: List[str] = dspy.OutputField(desc="List of tail entities for this head")


class ReflectionRevisionSignature(dspy.Signature):
    """Review and filter extracted triples for quality and validity."""
    triples: List[Dict[str, str]] = dspy.InputField(desc="List of extracted triples to review")
    valid_triples: List[Dict[str, str]] = dspy.OutputField(desc="Filtered list of valid triples only")


class DSPyNERExpert:
    """DSPy-based Named Entity Recognition expert."""
    
    def __init__(self):
        # DSPy LM configured globally - just create predictors
        self.extractor = dspy.Predict(EntityExtractionSignature)
    
    def extract_entities_by_type(self, text: str, entity_type: str, 
                                ontology: FinancialOntology) -> List[Entity]:
        """Extract entities using DSPy structured outputs."""
        try:
            with spinner(f"Extracting {entity_type}"):
                result = self.extractor(text=text, entity_type=entity_type)
            
            entities = []
            for entity_text in result.entities:
                if entity_text and entity_text.strip():
                    entities.append(Entity(
                        text=entity_text.strip(),
                        entity_type=entity_type
                    ))
            
            return entities
            
        except Exception as e:
            print(f"Error in DSPy NER extraction for {entity_type}: {e}")
            return []


class DSPyREExpert:
    """DSPy-based Relation Extraction expert."""
    
    def __init__(self, optimize=False, train_examples=None, val_examples=None):
        # DSPy LM configured globally - just create predictors
        self.head_extractor = dspy.Predict(HeadExtractionSignature)
        self.tail_extractor = dspy.Predict(TailExtractionSignature)
        if optimize and train_examples and val_examples:
            optimizer = DSPyOptimizer()
            self.tail_extractor = optimizer.optimize_predictor(
                self.tail_extractor, train_examples, val_examples
            )
            print(f"[DSPy OPT] ✅ Using optimized predictor with proper train/val split")
    
    def extract_relations_by_type(self, text: str, relation_type: str, 
                                 entities: List[Entity], ontology: FinancialOntology) -> Tuple[List[Triple], Set[str]]:
        """Extract relations using DSPy structured outputs."""
        if relation_type not in ontology.relation_types:
            return [], set()
        
        relation_info = ontology.relation_types[relation_type]
        head_types = set(relation_info.head_types)
        tail_types = set(relation_info.tail_types)
        
        head_entities = [e for e in entities if e.entity_type in head_types]
        tail_entities = [e for e in entities if e.entity_type in tail_types]
        
        triples = []
        missing_entities = set()
        
        try:
            # Step 1: Extract head candidates
            head_result = self.head_extractor(text=text, relation_type=relation_type)
            head_candidates = head_result.head_entities
            
            elapsed = get_elapsed_time()
            print_progress(f"[{elapsed:.1f}s] Found {len(head_candidates)} head candidates for {relation_type}")
            
            # Step 2: For each head, extract tails
            for i, head_entity in enumerate(head_candidates, 1):
                elapsed = get_elapsed_time()
                print_progress(f"[{elapsed:.1f}s] Processing head {i}/{len(head_candidates)}: {head_entity}")
                
                # Emergency timeout protection
                if elapsed > 3600:
                    print(f"\nEMERGENCY TIMEOUT: Stopping relation extraction after 1 hour")
                    break
                
                # Check if head entity exists in NER results
                matching_heads = [e for e in head_entities if e.text.lower() in head_entity.lower() 
                                 or head_entity.lower() in e.text.lower()]
                
                if not matching_heads:
                    missing_entities.add(head_entity)
                    continue
                
                # Extract tail entities with timeout protection
                try:
                    @with_timeout(15)  # 15 second timeout per call
                    def safe_tail_extraction():
                        return self.tail_extractor(
                            text=text, 
                            head_entity=head_entity, 
                            relation_type=relation_type
                        )
                    
                    tail_result = safe_tail_extraction()
                except TimeoutError:
                    print(f"⚠️ Timeout on head '{head_entity}' for relation '{relation_type}' - skipping")
                    continue
                except Exception as e:
                    print(f"⚠️ Error on head '{head_entity}' for relation '{relation_type}': {e} - skipping")
                    continue
                
                for tail_entity in tail_result.tail_entities:
                    # Check if tail entity exists in NER results
                    matching_tails = [e for e in tail_entities if e.text.lower() in tail_entity.lower()
                                     or tail_entity.lower() in e.text.lower()]
                    
                    if not matching_tails:
                        missing_entities.add(tail_entity)
                        continue
                    
                    # Validate relation before adding
                    head_text = matching_heads[0].text
                    tail_text = matching_tails[0].text
                    
                    # Skip self-referential relations
                    if head_text.lower().strip() != tail_text.lower().strip():
                        triples.append(Triple(
                            head=head_text,
                            relation=relation_type,
                            tail=tail_text
                        ))
        
        except Exception as e:
            print(f"Error in DSPy RE extraction for {relation_type}: {e}")
        
        return triples, missing_entities


class KnowledgeExtractionExpert:
    """DSPy-based Knowledge Extraction Expert with same interface as YAML version."""
    
    def __init__(self, model_name: str = "openai/gpt-3.5-turbo", prompts: str = "default", optimize=False, train_examples=None, val_examples=None):
        # DSPy LM configured globally - just create experts and predictors
        # Note: DSPy agents use signatures instead of prompts, but we accept the parameter for API consistency
        self.model_name = model_name
        self.prompts = prompts
        self.ner_expert = DSPyNERExpert()
        self.re_expert = DSPyREExpert(optimize=optimize, train_examples=train_examples, val_examples=val_examples)
        self.reflection_reviewer = dspy.Predict(ReflectionRevisionSignature)

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Main entry point - same interface as YAML version."""
        text = state.get("raw_text", "")
        ontology = state.get("ontology")
        
        if not text or not ontology:
            return {
                "error_log": state.get("error_log", []) + [
                    {"error": "Missing text or ontology for extraction"}
                ]
            }
        
        # Step 1: NER pass
        all_entities = self._extract_all_entities(text, ontology)
        elapsed = get_elapsed_time()
        print_progress(f"[{elapsed:.1f}s] DSPy NER completed: {len(all_entities)} entities", final=True)
        
        # Step 2: RE pass
        all_triples, _ = self._extract_relations_with_feedback(text, all_entities, ontology)
        elapsed = get_elapsed_time()
        print_progress(f"[{elapsed:.1f}s] DSPy RE completed: {len(all_triples)} triples", final=True)
        
        # Step 3: Apply reflection revision (DSPy version of YAML reflection)
        reflection_triples = self._apply_reflection_revision(all_triples)
        elapsed = get_elapsed_time()
        print_progress(f"[{elapsed:.1f}s] DSPy Reflection completed: {len(reflection_triples)} triples", final=True)
        
        # Step 4: Apply internal revision
        validated_entities, validated_triples = self._apply_internal_revision(all_entities, reflection_triples, ontology)
        print(f"\n[DSPy REVISION] Validated: {len(validated_entities)}/{len(all_entities)} entities, {len(validated_triples)}/{len(all_triples)} triples")
        
        # Update state with final results
        return {
            "extracted_entities": all_entities,
            "initial_triples": all_triples,
            "revised_entities": validated_entities,
            "revised_triples": validated_triples,
            "messages": state.get("messages", []) + [
                {
                    "agent": "dspy_knowledge_extraction_expert",
                    "action": "extraction_completed",
                    "entities_extracted": len(validated_entities),
                    "triples_extracted": len(validated_triples)
                }
            ]
        }
    
    def _extract_all_entities(self, text: str, ontology: FinancialOntology) -> List[Entity]:
        """Extract all entities by iterating through each entity type."""
        all_entities = []
        
        for entity_type in ontology.entity_types:
            entities = self.ner_expert.extract_entities_by_type(text, entity_type, ontology)
            all_entities.extend(entities)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in all_entities:
            entity_key = entity.text.lower().strip()
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_relations_with_feedback(self, text: str, entities: List[Entity], 
                                        ontology: FinancialOntology) -> Tuple[List[Triple], Set[str]]:
        """Extract relations with bidirectional feedback."""
        all_triples = []
        all_missing_entities = set()
        
        for i, relation_type in enumerate(ontology.relation_types, 1):
            elapsed = get_elapsed_time()
            print_progress(f"[{elapsed:.1f}s] Processing relation {i}/{len(ontology.relation_types)}: {relation_type}")
            triples, missing_entities = self.re_expert.extract_relations_by_type(
                text, relation_type, entities, ontology
            )
            all_triples.extend(triples)
            all_missing_entities.update(missing_entities)
        
        # Remove duplicates from triples
        seen = set()
        unique_triples = []
        for triple in all_triples:
            if hasattr(triple, 'head') and hasattr(triple, 'relation') and hasattr(triple, 'tail'):
                triple_key = (triple.head.lower(), triple.relation, triple.tail.lower())
                if triple_key not in seen:
                    seen.add(triple_key)
                    unique_triples.append(triple)
        
        return unique_triples, all_missing_entities
    
    def _apply_reflection_revision(self, triples: List[Triple]) -> List[Triple]:
        """Apply DSPy reflection revision to filter invalid triples."""
        if not triples:
            return []
        
        # Convert triples to dict format for DSPy processing
        triples_json = [{"head": t.head, "relation": t.relation, "tail": t.tail} for t in triples]
        
        print(f"[DSPy REFLECTION] Before: {len(triples)} triples")
        print(f"[DSPy REFLECTION] Before triplets: {[(t.head, t.relation, t.tail) for t in triples[:5]]}...")
        
        try:
            # Use DSPy for structured reflection
            result = self.reflection_reviewer(triples=triples_json)
            
            # Convert back to Triple objects
            reflection_triples = []
            for triple_data in result.valid_triples:
                if (isinstance(triple_data, dict) and triple_data.get("head") and 
                    triple_data.get("relation") and triple_data.get("tail")):
                    reflection_triples.append(Triple(
                        head=triple_data["head"],
                        relation=triple_data["relation"], 
                        tail=triple_data["tail"]
                    ))
            
            print(f"[DSPy REFLECTION] After: {len(reflection_triples)} triples")
            print(f"[DSPy REFLECTION] After triplets: {[(t.head, t.relation, t.tail) for t in reflection_triples[:5]]}...")
            print(f"[DSPy REFLECTION] Filtered out: {len(triples) - len(reflection_triples)} invalid triples")
            
            return reflection_triples
            
        except Exception as e:
            print(f"[DSPy REFLECTION] Failed: {e}")
            return triples  # Fallback to original triples
    
    def _apply_internal_revision(self, entities: List[Entity], triples: List[Triple], 
                                ontology: FinancialOntology) -> Tuple[List[Entity], List[Triple]]:
        """Apply internal revision rules."""
        # Validate entities
        valid_types = set(ontology.entity_types.keys())
        validated_entities = [e for e in entities if e.entity_type in valid_types and e.text.strip()]
        
        # Validate triples with fuzzy entity matching
        entity_texts = {e.text.lower().strip() for e in validated_entities}
        valid_relations = set(ontology.relation_types.keys())
        
        def entity_exists(text):
            text_clean = text.lower().strip()
            if text_clean in entity_texts:
                return True
            for entity_text in entity_texts:
                if text_clean in entity_text or entity_text in text_clean:
                    if abs(len(text_clean) - len(entity_text)) <= 3:
                        return True
            return False
        
        validated_triples = []
        for triple in triples:
            if (triple.relation in valid_relations and 
                entity_exists(triple.head) and 
                entity_exists(triple.tail) and
                triple.head.strip() and triple.tail.strip()):
                validated_triples.append(triple)
        
        return validated_entities, validated_triples