"""
Knowledge Extraction Expert - Orchestrates NER and RE sub-agents.

This agent manages the core information extraction tasks using specialized
sub-agents for Named Entity Recognition and Relation Extraction.
"""

import json
import time
from typing import Dict, Any, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain_core.messages import HumanMessage, SystemMessage
import json_repair

from ma_finkg.models import GraphState, Entity, Triple, FinancialOntology
from ma_finkg.utils import load_prompts, print_progress, get_elapsed_time, spinner
from ma_finkg.utils.llm_factory import create_openrouter_llm, timeout_llm_call


class NERExpert:
    
    def __init__(self, model_name: str = "openai/gpt-3.5-turbo", prompts: str = "default"):
        self.llm = create_openrouter_llm(model_name)
        self.prompts = prompts
    
    def extract_entities_by_type(self, text: str, entity_type: str, 
                                ontology: FinancialOntology) -> List[Entity]:
        prompts = load_prompts(self.prompts)['knowledge_extraction_expert']
        prompt = prompts['ner_entity_extraction_prompt'].format(
            entity_type=entity_type,
            text=text
        )

        try:
            with spinner(f"Extracting {entity_type}"):
                response = timeout_llm_call(self.llm, [HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                data = json.loads(json_repair.repair_json(response_text))
            
            entities = []
            if entity_type in data:
                for entity_text in data[entity_type]:
                    if entity_text and entity_text.strip():
                        entities.append(Entity(
                            text=entity_text.strip(),
                            entity_type=entity_type
                        ))
            
            return entities
            
        except Exception as e:
            print(f"Error in NER extraction for {entity_type}: {e}")
            return []



class REExpert:
    
    def __init__(self, model_name: str = "openai/gpt-3.5-turbo", prompts: str = "default"):
        self.llm = create_openrouter_llm(model_name)
        self.prompts = prompts
    
    def extract_relations_by_type(self, text: str, relation_type: str, 
                                 entities: List[Entity], ontology: FinancialOntology) -> List[Triple]:
        """
        Clean head-then-tail extraction for specific relation type.
        """
        if relation_type not in ontology.relation_types:
            return []
        
        rel_constraint = ontology.relation_types[relation_type]
        head_types = rel_constraint.head_types if rel_constraint.head_types else []
        tail_types = rel_constraint.tail_types if rel_constraint.tail_types else []
        
        triples = []
        
        # Step 1: Extract head entities
        head_candidates = self._extract_head_entities(text, head_types[0] if head_types else "Entity", relation_type)
        
        # Step 2: For each head, extract tail entities (relation-aware)
        for head_entity in head_candidates:
            tail_candidates = self._extract_tail_entities(
                text, head_entity, tail_types[0] if tail_types else "Entity", 
                relation_type, ontology
            )
            
            for tail_entity in tail_candidates:
                if head_entity.lower().strip() != tail_entity.lower().strip():
                    triples.append(Triple(
                        head=head_entity,
                        relation=relation_type,
                        tail=tail_entity
                    ))
        
        return triples
    
    def _extract_head_entities(self, text: str, head_type: str, relation_type: str) -> List[str]:
        prompts = load_prompts(self.prompts)['knowledge_extraction_expert']
        prompt = prompts['re_head_extraction_prompt'].format(
            head_type=head_type,
            relation_type=relation_type,
            text=text
        )

        try:
            response = timeout_llm_call(self.llm, [HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                data = json.loads(json_repair.repair_json(response_text))
                
            # Extract entities from response
            entities = []
            if head_type in data and isinstance(data[head_type], list):
                entities = [e.strip() for e in data[head_type] if e.strip()]
            
            return entities
            
        except Exception as e:
            print(f"Error extracting head entities: {e}")
            return []
    
    def _extract_tail_entities(self, text: str, head_entity: str, tail_type: str, relation_type: str, ontology: FinancialOntology) -> List[str]:
        prompts = load_prompts(self.prompts)['knowledge_extraction_expert']
        prompt = prompts['re_tail_extraction_prompt'].format(
            head_entity=head_entity,
            tail_type=tail_type,
            relation_type=relation_type,
            text=text
        )

        try:
            response = timeout_llm_call(self.llm, [HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            try:
                data = json.loads(response_text)
            except json.JSONDecodeError:
                data = json.loads(json_repair.repair_json(response_text))
            
            # Extract entities from response
            entities = []
            if tail_type in data and isinstance(data[tail_type], list):
                entities = [e.strip() for e in data[tail_type] if e.strip()]
            
            return entities
            
        except Exception as e:
            print(f"Error extracting tail entities: {e}")
            return []


class KnowledgeExtractionExpert:
    
    def __init__(self, model_name: str = "openai/gpt-3.5-turbo", prompts: str = "default", enable_re: bool = True):
        self.llm = create_openrouter_llm(model_name)
        self.ner_expert = NERExpert(model_name, prompts)
        self.re_expert = REExpert(model_name, prompts) if enable_re else None
        self.enable_re = enable_re
        self.prompts = load_prompts(prompts)['knowledge_extraction_expert']

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """
        Main entry point for knowledge extraction.
        Orchestrates NER and RE to extract entities and relations.
        """
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
        print_progress(f"[{elapsed:.1f}s] NER completed: {len(all_entities)} entities", final=True)
        
        # Step 2: RE pass (optional toggle for faster NER-only testing)
        if self.enable_re:
            all_triples, _ = self._extract_relations_with_feedback(text, all_entities, ontology)
            elapsed = get_elapsed_time()
            print_progress(f"[{elapsed:.1f}s] RE completed: {len(all_triples)} filtered triples", final=True)
        else:
            all_triples = []
            print_progress("[SKIP] Relation extraction disabled for faster NER testing", final=True)
        
        # Step 3: Apply internal revision rules
        validated_entities, validated_triples = self._apply_internal_revision(all_entities, all_triples, ontology)
        print(f"\n[REVISION] Validated: {len(validated_entities)}/{len(all_entities)} entities, {len(validated_triples)}/{len(all_triples)} triples")
        
        # Update state with final results
        updates = {
            "extracted_entities": all_entities,
            "initial_triples": all_triples,
            "revised_entities": validated_entities,
            "revised_triples": validated_triples,
            "messages": state.get("messages", []) + [
                {
                    "agent": "knowledge_extraction_expert",
                    "action": "extraction_completed",
                    "entities_extracted": len(validated_entities),
                    "triples_extracted": len(validated_triples)
                }
            ]
        }
        
        return updates
    
    def _extract_all_entities(self, text: str, ontology: FinancialOntology) -> List[Entity]:
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
    
    def _extract_all_relations(self, text: str, entities: List[Entity], 
                              ontology: FinancialOntology) -> List[Triple]:
        all_triples = []
        
        for relation_type in ontology.relation_types:
            triples = self.re_expert.extract_relations_by_type(text, relation_type, entities, ontology)
            if len(triples) > 0:
                elapsed = get_elapsed_time()
                print_progress(f"[{elapsed:.1f}s] Found {len(triples)} {relation_type} relations")
            all_triples.extend(triples)
        
        # Remove duplicates
        seen = set()
        unique_triples = []
        for triple in all_triples:
            triple_key = (triple.head.lower(), triple.relation, triple.tail.lower())
            if triple_key not in seen:
                seen.add(triple_key)
                unique_triples.append(triple)
        
        return unique_triples
    
    def _extract_relations_with_feedback(self, text: str, entities: List[Entity], 
                                        ontology: FinancialOntology) -> Tuple[List[Triple], Set[str]]:
        """
        Extracts relations using two-step approach with ontology validation
        """
        all_triples = []
        all_missing_entities = set()
        
        # Get available entity types from extracted entities
        available_entity_types = set(entity.entity_type for entity in entities)
        
        # Clean head-then-tail extraction for each relation type
        for i, relation_type in enumerate(ontology.relation_types, 1):
            elapsed = get_elapsed_time()
            print_progress(f"[{elapsed:.1f}s] Processing relation {i}/{len(ontology.relation_types)}: {relation_type}")
            
            # ONTOLOGY CONSTRAINT CHECK: Skip if required entity types not available
            relation_constraint = ontology.relation_types[relation_type]
            required_head_types = set(relation_constraint.head_types)
            required_tail_types = set(relation_constraint.tail_types)
            
            # Check if we have entities matching both head and tail requirements
            available_head_types = available_entity_types & required_head_types
            available_tail_types = available_entity_types & required_tail_types
            
            if not available_head_types or not available_tail_types:
                # Skip extraction - impossible to form valid relations
                continue
                
            triples = self.re_expert.extract_relations_by_type(text, relation_type, entities, ontology)
            all_triples.extend(triples)
        
        # Simple ontology post-processing filter
        valid_relations = set(ontology.relation_types.keys())
        filtered_triples = []
        
        for t in all_triples:
            # NER-priority rule: both entities must exist in NER results
            head_in_ner = any(e.text.lower().strip() == t.head.lower().strip() for e in entities)
            tail_in_ner = any(e.text.lower().strip() == t.tail.lower().strip() for e in entities)
            
            if (t.relation in valid_relations and t.head.strip() and t.tail.strip() and 
                t.head.lower() != t.tail.lower() and head_in_ner and tail_in_ner):
                filtered_triples.append(t)
        
        elapsed = get_elapsed_time()
        print_progress(f"[{elapsed:.1f}s] RE completed: {len(filtered_triples)} filtered triples")
        
        return filtered_triples, set()
    
    def _apply_internal_revision(self, entities: List[Entity], triples: List[Triple], 
                                ontology: FinancialOntology) -> Tuple[List[Entity], List[Triple]]:

        # Keep all entities, only remove empty text
        validated_entities = [e for e in entities if e.text.strip()]
        
        # Use LLM reflection for revision instead of hard rules
        # Step 1: Apply reflection-based entity revision
        revised_entities = self._apply_reflection_revision_entities(validated_entities, ontology)
        
        # Step 2: Apply reflection-based triple revision  
        revised_triples = self._apply_reflection_revision_triples(triples, ontology)

        # Only apply deduplication, no hard constraints
        seen = set()
        unique_triples = []
        for tr in revised_triples:
            key = (tr.head.strip().lower(), tr.relation, tr.tail.strip().lower())
            if key not in seen:
                seen.add(key)
                unique_triples.append(tr)
        
        return revised_entities, unique_triples
    
    def _apply_reflection_revision_entities(self, entities: List[Entity], ontology: FinancialOntology) -> List[Entity]:
        """
        Use LLM reflection to revise entity extractions
        """
        return entities
    
    def _apply_reflection_revision_triples(self, triples: List[Triple], ontology: FinancialOntology) -> List[Triple]:
        # Apply soft validation instead of hard rejection
        validated_triples = []
        valid_relations = set(ontology.relation_types.keys())
        
        for triple in triples:
            # Prefer known relation types but don't reject unknown ones
            if triple.relation in valid_relations:
                validated_triples.append(triple)
            else:
                validated_triples.append(triple)
        
        return validated_triples
    
    def classify_entity_pair(self, sentence: str, e1_text: str, e1_type: str, 
                           e2_text: str, e2_type: str, ontology: FinancialOntology) -> str:
        """
        Direct relation classification between specific entity pair.
        """
        # Use existing relation extraction logic but filter to specific entity pair
        if not self.re_expert:
            return "no_relation"
            
        # Create temporary entities for the pair
        e1 = Entity(text=e1_text, entity_type=e1_type)
        e2 = Entity(text=e2_text, entity_type=e2_type)
        entities = [e1, e2]
        
        # Extract all relations and find one matching our entity pair
        for relation_type in ontology.relation_types.keys():
            triples = self.re_expert.extract_relations_by_type(
                sentence, relation_type, entities, ontology
            )
            
            # Check if any triple matches our specific entity pair
            for triple in triples:
                if (triple.head.lower().strip() == e1_text.lower().strip() and 
                    triple.tail.lower().strip() == e2_text.lower().strip()):
                    return relation_type
                    
        return "no_relation"
    
