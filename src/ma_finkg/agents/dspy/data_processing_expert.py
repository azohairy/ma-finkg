"""
DSPy-based Data Processing Expert - Handles data I/O and validation with structured outputs.

This agent provides the same interface as the YAML-based version but uses DSPy
for any text processing that requires LLM intervention.
"""

import dspy
import json
import re
from typing import Dict, Any, List, Optional

from ma_finkg.models import GraphState, Entity, Triple, ValidationError


class TextProcessingSignature(dspy.Signature):
    """Process and clean text for knowledge extraction."""
    raw_text: str = dspy.InputField(desc="Raw input text to process")
    processed_text: str = dspy.OutputField(desc="Cleaned and normalized text ready for extraction")


class DataProcessingExpert:
    """DSPy-based Data Processing Expert with same interface as YAML version."""
    
    def __init__(self):
        # Most data processing is rule-based, DSPy available for complex text processing
        self.text_processor = dspy.Predict(TextProcessingSignature)

    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Main entry point - same interface as YAML version."""
        raw_text = state.get("raw_text", "")
        
        # Validate and process the input text
        processed_text, validation_results = self._process_and_validate_text(raw_text)
        
        # Update processing statistics
        stats = self._generate_text_statistics(processed_text)
        
        updates = {
            "raw_text": processed_text,
            "processing_stats": {
                **state.get("processing_stats", {}),
                **stats,
                "data_validation": validation_results
            },
            "messages": state.get("messages", []) + [
                {
                    "agent": "dspy_data_processing_expert",
                    "action": "text_processed",
                    "stats": stats,
                    "validation": validation_results
                }
            ]
        }
        
        return updates
    
    def _process_and_validate_text(self, text: str) -> tuple[str, Dict[str, Any]]:
        """Process and validate input text - same logic as YAML version."""
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check for empty or too short text
        if not text or len(text.strip()) < 50:
            validation_results["errors"].append("Text too short for meaningful extraction")
            validation_results["is_valid"] = False
        
        # Clean and normalize text (rule-based, no DSPy needed)
        processed_text = self._clean_text(text)
        
        # Check text quality metrics
        if len(processed_text.split()) < 20:
            validation_results["warnings"].append("Text may be too short for comprehensive extraction")
        
        return processed_text, validation_results
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text - rule-based processing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove or normalize special characters that might interfere with JSON
        text = text.replace('\x00', '').replace('\r', ' ')
        
        return text
    
    def _generate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Generate text statistics - rule-based processing."""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "text_length": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "avg_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
    
    def validate_json_output(self, json_text: str, expected_keys: List[str]) -> Dict[str, Any]:
        """Validate JSON output - rule-based validation."""
        validation_result = {
            "is_valid": False,
            "data": None,
            "errors": []
        }
        
        try:
            data = json.loads(json_text)
            validation_result["data"] = data
            
            # Check for expected keys
            missing_keys = [key for key in expected_keys if key not in data]
            if missing_keys:
                validation_result["errors"].append(f"Missing required keys: {missing_keys}")
            else:
                validation_result["is_valid"] = True
                
        except json.JSONDecodeError as e:
            validation_result["errors"].append(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            validation_result["errors"].append(f"Unexpected error: {str(e)}")
        
        return validation_result
    
    def store_entities(self, entities: List[Entity], state: GraphState) -> Dict[str, Any]:
        """Store and validate extracted entities - rule-based processing."""
        # Validate entity format and content
        validated_entities = []
        validation_errors = []
        
        for entity in entities:
            if self._validate_entity(entity):
                validated_entities.append(entity)
            else:
                error = ValidationError(
                    error_type="invalid_entity",
                    description=f"Entity validation failed: {entity.text}",
                    entity=entity
                )
                validation_errors.append(error)
        
        return {
            "extracted_entities": validated_entities,
            "error_log": state.get("error_log", []) + validation_errors,
            "messages": state.get("messages", []) + [
                {
                    "agent": "dspy_data_processing_expert",
                    "action": "entities_stored",
                    "count": len(validated_entities),
                    "errors": len(validation_errors)
                }
            ]
        }
    
    def store_triples(self, triples: List[Triple], state: GraphState, 
                     triple_type: str = "initial") -> Dict[str, Any]:
        """Store and validate knowledge graph triples - rule-based processing."""
        # Validate triple format
        validated_triples = []
        validation_errors = []
        
        for triple in triples:
            if self._validate_triple(triple):
                validated_triples.append(triple)
            else:
                error = ValidationError(
                    error_type="invalid_triple",
                    description=f"Triple validation failed: {triple.head} -> {triple.relation} -> {triple.tail}",
                    triple=triple
                )
                validation_errors.append(error)
        
        # Store in appropriate field based on type
        field_name = f"{triple_type}_triples"
        
        return {
            field_name: validated_triples,
            "error_log": state.get("error_log", []) + validation_errors,
            "messages": state.get("messages", []) + [
                {
                    "agent": "dspy_data_processing_expert", 
                    "action": f"{triple_type}_triples_stored",
                    "count": len(validated_triples),
                    "errors": len(validation_errors)
                }
            ]
        }
    
    def _validate_entity(self, entity: Entity) -> bool:
        """Validate individual entity - rule-based validation."""
        if not entity.text or not entity.text.strip():
            return False
        
        if not entity.entity_type or not entity.entity_type.strip():
            return False
        
        # Check for reasonable entity text length
        if len(entity.text) > 200:  # Entities shouldn't be too long
            return False
        
        return True
    
    def _validate_triple(self, triple: Triple) -> bool:
        """Validate individual triple - rule-based validation."""
        if not all([triple.head, triple.relation, triple.tail]):
            return False
        
        if not all([triple.head.strip(), triple.relation.strip(), triple.tail.strip()]):
            return False
        
        # Check for reasonable text lengths
        if any(len(field) > 200 for field in [triple.head, triple.relation, triple.tail]):
            return False
        
        return True
    
    def export_final_kg(self, state: GraphState) -> Dict[str, Any]:
        """Export final knowledge graph - rule-based processing."""
        entities = state.get("revised_entities", [])
        triples = state.get("revised_triples", [])
        
        # Create export data
        export_data = {
            "metadata": {
                "total_entities": len(entities),
                "total_triples": len(triples),
                "entity_types": list(set(e.entity_type for e in entities)),
                "relation_types": list(set(t.relation for t in triples)),
                "processing_stats": state.get("processing_stats", {})
            },
            "entities": [
                {
                    "text": e.text,
                    "type": e.entity_type
                }
                for e in entities
            ],
            "triples": [
                {
                    "head": t.head,
                    "relation": t.relation, 
                    "tail": t.tail
                }
                for t in triples
            ]
        }
        
        return {
            "final_kg_export": export_data,
            "messages": state.get("messages", []) + [
                {
                    "agent": "dspy_data_processing_expert",
                    "action": "kg_exported",
                    "entities": len(entities),
                    "triples": len(triples)
                }
            ]
        }
    
    def _dspy_process_complex_text(self, text: str) -> str:
        """DSPy-based complex text processing when needed."""
        try:
            result = self.text_processor(raw_text=text)
            return result.processed_text
        except Exception as e:
            print(f"[DSPy DATA] Complex processing failed: {e}, using rule-based fallback")
            return self._clean_text(text)