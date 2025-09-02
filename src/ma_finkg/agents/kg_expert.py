from typing import Dict, Any
from langchain_core.messages import HumanMessage
from ma_finkg.models import GraphState
from ma_finkg.utils import load_prompts
from ma_finkg.utils.llm_factory import create_openrouter_llm, timeout_llm_call

class KnowledgeGraphExpert:
    def __init__(self, model_name: str = "openai/gpt-3.5-turbo", prompts: str = "default"):
        self.llm = create_openrouter_llm(model_name)
        self.prompts = load_prompts(prompts)['kg_expert']
    
    def __call__(self, state: GraphState) -> Dict[str, Any]:
        next_action, reasoning = self._intelligent_coordination(state)
        return {
            "next_agent": next_action,
            "messages": state.get("messages", []) + [{"agent": "kg_expert", "action": next_action, "reasoning": reasoning}]
        }
    
    def _intelligent_coordination(self, state: GraphState) -> tuple[str, str]:
        """Intelligent coordination with reasoning"""
        
        # Simple safety check - fallback to rule-based if needed
        if len(state.get("messages", [])) > 15:
            return "finalize", "Maximum steps reached"
        
        has_ontology = bool(state.get("ontology"))
        has_entities = len(state.get("revised_entities", [])) > 0
        has_triples = len(state.get("revised_triples", [])) > 0
        extraction_attempted = any(msg.get("agent") == "knowledge_extraction_expert" for msg in state.get("messages", []))
        prompt = self.prompts['coordination_prompt'].format(
            system_prompt=self.prompts['system_prompt'],
            has_ontology=has_ontology,
            has_entities=has_entities,
            entity_count=len(state.get("revised_entities", [])),
            has_triples=has_triples,
            triple_count=len(state.get("revised_triples", [])),
            extraction_attempted=extraction_attempted,
            step_count=len(state.get("messages", []))
        )
        
        try:
            response = timeout_llm_call(self.llm, [HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            if ":" in response_text:
                agent, reasoning = response_text.split(":", 1)
                return agent.strip(), reasoning.strip()
            else:
                # Fall through to rule-based logic
                pass
                    
        except Exception:
            pass  # Fall back to rule-based
        
        # Rule-based coordination (used as fallback or when LLM parsing fails)
        if not has_ontology:
            return "domain_specific_expert", "Ontology needed"
        elif not extraction_attempted:
            return "knowledge_extraction_expert", "Extraction needed"
        else:
            return "finalize", "Extraction completed, ready to finalize"
    
    def initialize_state(self, raw_text: str) -> Dict[str, Any]:
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
        return {
            "next_agent": "END",
            "revised_entities": state.get("revised_entities", []),
            "revised_triples": state.get("revised_triples", []),
            "messages": state.get("messages", []) + [{"agent": "kg_expert", "action": "finalize"}]
        }