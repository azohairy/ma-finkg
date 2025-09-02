"""
DSPy-based Knowledge Graph Expert - Coordinates multi-agent workflow with structured outputs.

This agent provides the same interface as the YAML-based version but uses DSPy
for reliable coordination decisions without parsing errors.
"""

import dspy
from typing import Dict, Any

from ma_finkg.models import GraphState


class CoordinationDecisionSignature(dspy.Signature):
    """Make intelligent coordination decisions for multi-agent workflow."""
    
    # System context
    system_context: str = dspy.InputField(desc="System context and available agents")
    
    # Current state information
    has_ontology: bool = dspy.InputField(desc="Whether ontology exists")
    has_entities: bool = dspy.InputField(desc="Whether entities have been extracted")
    entity_count: int = dspy.InputField(desc="Number of extracted entities")
    has_triples: bool = dspy.InputField(desc="Whether triples have been extracted")
    triple_count: int = dspy.InputField(desc="Number of extracted triples")
    extraction_attempted: bool = dspy.InputField(desc="Whether knowledge extraction was attempted")
    step_count: int = dspy.InputField(desc="Number of processing steps taken")
    
    # Structured output
    next_agent: str = dspy.OutputField(desc="Next agent to call (domain_specific_expert, data_processing_expert, knowledge_extraction_expert, or finalize)")
    reasoning: str = dspy.OutputField(desc="Reasoning for this decision")


class KnowledgeGraphExpert:
    """DSPy-based Knowledge Graph Expert with same interface as YAML version."""
    
    def __init__(self, model_name: str = "openai/gpt-3.5-turbo", prompts: str = "default"):
        # DSPy LM configured globally - just create predictors
        # Note: DSPy agents use signatures instead of prompts, but we accept the parameter for API consistency
        self.model_name = model_name
        self.prompts = prompts
        self.coordinator = dspy.Predict(CoordinationDecisionSignature)
    
    def __call__(self, state: GraphState) -> Dict[str, Any]:
        """Main entry point - same interface as YAML version."""
        next_action, reasoning = self._intelligent_coordination(state)
        return {
            "next_agent": next_action,
            "messages": state.get("messages", []) + [
                {
                    "agent": "dspy_kg_expert", 
                    "action": next_action, 
                    "reasoning": reasoning
                }
            ]
        }
    
    def _intelligent_coordination(self, state: GraphState) -> tuple[str, str]:
        """DSPy-based intelligent coordination with structured outputs."""
        
        # Simple safety check - fallback to rule-based if needed
        if len(state.get("messages", [])) > 15:
            return "finalize", "Maximum steps reached"
        
        # Analyze current state
        has_ontology = bool(state.get("ontology"))
        has_entities = len(state.get("revised_entities", [])) > 0
        has_triples = len(state.get("revised_triples", [])) > 0
        extraction_attempted = any(
            msg.get("agent") in ["knowledge_extraction_expert", "dspy_knowledge_extraction_expert"] 
            for msg in state.get("messages", [])
        )
        
        # System context for DSPy
        system_context = """
        You coordinate a multi-agent KG construction system with these agents:
        - domain_specific_expert: Builds the ontology
        - data_processing_expert: Cleans and prepares text  
        - knowledge_extraction_expert: Extracts entities and relations
        - finalize: Completes the process
        
        Rules:
        - If ontology missing: call domain_specific_expert
        - If text not processed: call data_processing_expert  
        - If entities/relations not extracted: call knowledge_extraction_expert
        - If extraction attempted (even with 0 results): finalize
        """
        
        try:
            # Use DSPy for structured coordination decision
            result = self.coordinator(
                system_context=system_context,
                has_ontology=has_ontology,
                has_entities=has_entities,
                entity_count=len(state.get("revised_entities", [])),
                has_triples=has_triples,
                triple_count=len(state.get("revised_triples", [])),
                extraction_attempted=extraction_attempted,
                step_count=len(state.get("messages", []))
            )
            
            # Validate agent choice
            valid_agents = ["domain_specific_expert", "data_processing_expert", "knowledge_extraction_expert", "finalize"]
            if result.next_agent in valid_agents:
                return result.next_agent, result.reasoning
            else:
                print(f"[DSPy COORD] Invalid agent '{result.next_agent}', falling back to rule-based")
                return self._fallback_rule_based_coordination(state)
                
        except Exception as e:
            print(f"[DSPy COORD] DSPy coordination failed: {e}, using rule-based fallback")
            return self._fallback_rule_based_coordination(state)
    
    def _fallback_rule_based_coordination(self, state: GraphState) -> tuple[str, str]:
        """Fallback rule-based coordination logic."""
        has_ontology = bool(state.get("ontology"))
        has_entities = len(state.get("revised_entities", [])) > 0
        has_triples = len(state.get("revised_triples", [])) > 0
        extraction_attempted = any(
            msg.get("agent") in ["knowledge_extraction_expert", "dspy_knowledge_extraction_expert"] 
            for msg in state.get("messages", [])
        )
        
        # Rule-based logic
        if not has_ontology:
            return "domain_specific_expert", "No ontology created yet"
        elif not extraction_attempted:
            return "knowledge_extraction_expert", "Ready for knowledge extraction"
        else:
            return "finalize", "Extraction completed, ready to finalize"
    
    def initialize_state(self, raw_text: str) -> Dict[str, Any]:
        """Initialize state - same interface as YAML version."""
        return {
            "raw_text": raw_text,
            "ontology": None,
            "extracted_entities": [],
            "initial_triples": [],
            "revised_entities": [],
            "revised_triples": [],
            "next_agent": "domain_specific_expert",
            "messages": []
        }
    
    def finalize_kg_construction(self, state: GraphState) -> Dict[str, Any]:
        """Finalize KG construction - same interface as YAML version."""
        return {
            "next_agent": "END",
            "revised_entities": state.get("revised_entities", []),
            "revised_triples": state.get("revised_triples", []),
            "messages": state.get("messages", []) + [
                {"agent": "dspy_kg_expert", "action": "finalize"}
            ]
        }