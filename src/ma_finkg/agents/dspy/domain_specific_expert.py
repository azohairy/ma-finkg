"""
DSPy-based Domain Specific Expert - Creates ontologies with structured outputs.

This agent provides the same interface as the YAML-based version but uses DSPy
for reliable ontology creation without JSON parsing errors.
"""

import dspy
from typing import Dict, Any, List

from ma_finkg.models import GraphState, FinancialOntology, EntityType, RelationType
from ma_finkg.utils import spinner


class OntologyCreationSignature(dspy.Signature):
    """Create domain-specific ontology for knowledge extraction."""
    text: str = dspy.InputField(desc="Text to analyze for ontology creation")
    # Simple structured output matching existing conversion logic
    entity_types_with_examples: Dict[str, List[str]] = dspy.OutputField(desc="Entity types mapped to example lists")
    relation_types_with_constraints: Dict[str, Dict[str, str]] = dspy.OutputField(desc="Relations mapped to head/tail constraints")


class DomainSpecificExpert:
    """DSPy-based Domain Specific Expert with same interface as YAML version."""
    
    def __init__(self, ontology: str = "default"):
        # DSPy LM configured globally - just create predictors
        # Note: DSPy agents use signatures instead of prompts, but we accept the parameter for API consistency
        self.ontology_type = ontology
        self.ontology_creator = dspy.Predict(OntologyCreationSignature)
    
    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Main entry point - same interface as YAML version."""
        ontology = self._create_domain_ontology(state.get("raw_text", ""))
        return {
            "ontology": ontology,
            "messages": state.get("messages", []) + [
                {"agent": "dspy_domain_specific_expert", "action": "ontology_defined"}
            ]
        }
    
    def _create_domain_ontology(self, text: str = "") -> FinancialOntology:
        """Create ontology using predefined schema like main agent."""
        print(f"\n[DSPy ONTOLOGY] Using predefined {self.ontology_type} ontology")
        result = self._get_predefined_ontology(self.ontology_type)
        self._log_ontology(result)
        return result
    
    def _get_predefined_ontology(self, ontology: str = "default") -> FinancialOntology:
        """Load static ontology from predefined schema"""
        from ma_finkg.config import Config
        import json
        
        # Load predefined ontology schema
        ontology_path = Config.get_ontology_file(ontology=ontology)
        with open(ontology_path, 'r') as f:
            ontology_schema = json.load(f)
        
        # Build entity types
        entity_types = {}
        for name, examples in ontology_schema["entities"].items():
            entity_types[name] = EntityType(name=name, examples=examples)
        
        # Build relation types with list constraints (no single-type picking)
        relation_types = {}
        for name, rel_info in ontology_schema["relations"].items():
            relation_types[name] = RelationType(
                name=name,
                head_types=rel_info["head_types"],
                tail_types=rel_info["tail_types"]
            )
        
        return FinancialOntology(entity_types=entity_types, relation_types=relation_types)
    
    def _log_ontology(self, ontology: FinancialOntology):
        """Log the created ontology for debugging."""
        print(f"\n[DSPy ONTOLOGY] Entity Types: {list(ontology.entity_types.keys())}")
        print(f"[DSPy ONTOLOGY] Relation Types: {list(ontology.relation_types.keys())}")
        # for rel_name, rel_type in ontology.relation_types.items():
        #     print(f"[DSPy ONTOLOGY]   {rel_name}: {rel_type.head_types} -> {rel_type.tail_types}")
    
    def get_entity_types_list(self, ontology: FinancialOntology) -> list:
        """Returns a list of entity type names for extraction tasks."""
        return list(ontology.entity_types.keys())
    
    def get_relation_types_list(self, ontology: FinancialOntology) -> list:
        """Returns a list of relation type names for extraction tasks."""
        return list(ontology.relation_types.keys())
    
    def validate_entity_type(self, entity_type: str, ontology: FinancialOntology) -> bool:
        """Validates if an entity type exists in the ontology."""
        return entity_type in ontology.entity_types
    
    def validate_relation_constraints(self, relation: str, head_type: str, tail_type: str, 
                                    ontology: FinancialOntology) -> bool:
        """Validates if a relation satisfies the head/tail type constraints."""
        if relation not in ontology.relation_types:
            return False
        
        rel_type = ontology.relation_types[relation]
        return (head_type in rel_type.head_types and tail_type in rel_type.tail_types)
    
    def _collaborative_ontology_creation(self, text: str) -> FinancialOntology:
        """DSPy-based ontology creation with structured outputs."""
        ontology_data = self._dspy_domain_expert_provide_ontology(text)
        return self._build_ontology_from_collaboration(ontology_data)
    
    def _dspy_domain_expert_provide_ontology(self, text: str) -> Dict:
        """DSPy-based ontology creation - outputs format matching existing conversion logic."""
        with spinner("Creating DSPy ontology"):
            result = self.ontology_creator(text=text)
        
        print(f"[DSPy ONTOLOGY] Structured response received")
        
        # Format data to match existing _build_ontology_from_collaboration expectations
        return {
            "entities": result.entity_types_with_examples,
            "relations": result.relation_types_with_constraints
        }
    
    def _build_ontology_from_collaboration(self, data: Dict) -> FinancialOntology:
        """Build ontology from DSPy structured response."""
        entity_types = {}
        for name, examples in data.get("entities", {}).items():
            entity_types[name] = EntityType(name=name, examples=examples)
        
        relation_types = {}
        for name, rel_info in data.get("relations", {}).items():
            # Defensive handling: convert to list format if needed
            head_types = rel_info.get("head", [])
            tail_types = rel_info.get("tail", [])
            
            if isinstance(head_types, str):
                head_types = [head_types] if head_types else []
            if isinstance(tail_types, str):
                tail_types = [tail_types] if tail_types else []
                
            relation_types[name] = RelationType(
                name=name,
                head_types=head_types,
                tail_types=tail_types
            )
        
        return FinancialOntology(entity_types=entity_types, relation_types=relation_types)