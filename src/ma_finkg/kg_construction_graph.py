from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ma_finkg.models import GraphState
from ma_finkg.agents import (
    KnowledgeGraphExpert,
    DomainSpecificExpert,
    DataProcessingExpert, 
    KnowledgeExtractionExpert
)
from ma_finkg.utils import set_global_timer, get_elapsed_time, chunk_large_text


class FinancialKGConstructionGraph:
    """
    Main orchestrator that builds and manages the multi-agent collaborative
    framework for financial knowledge graph construction using LangGraph.
    """

    def __init__(self, model_name: str = "openai/gpt-3.5-turbo", 
                 ontology: str = "default", 
                 prompts: str = "default",
                 use_dspy: bool = False, 
                 dspy_eval_examples=None, dspy_train_examples=None, dspy_val_examples=None,
                 enable_re: bool = True):
        """
        Initialize the multi-agent system with all specialized agents.
        """
        self.model_name = model_name
        self.ontology = ontology  
        self.prompts = prompts
        
        # Switch between YAML and DSPy agents
        if use_dspy:
            # Apply OpenAI/LiteLLM compatibility fix before importing DSPy
            try:
                import openai.types.responses.response as response_module
                import openai.types.responses.response_create_params as params_module
                
                # Fix ResponseTextConfig -> ResponseFormatTextConfig
                if not hasattr(response_module, 'ResponseTextConfig'):
                    response_module.ResponseTextConfig = response_module.ResponseFormatTextConfig
                    
                # Fix ResponseTextConfigParam -> ResponseFormatTextConfigParam  
                if not hasattr(params_module, 'ResponseTextConfigParam'):
                    params_module.ResponseTextConfigParam = params_module.ResponseFormatTextConfigParam
            except (ImportError, AttributeError):
                pass
            
            import dspy
            from ma_finkg.utils.llm_factory import create_dspy_llm
            
            # Configure DSPy once globally
            dspy_llm = create_dspy_llm(model_name)
            dspy.settings.configure(lm=dspy_llm)
            
            from ma_finkg.agents.dspy.kg_expert import KnowledgeGraphExpert as DSPyKnowledgeGraphExpert
            from ma_finkg.agents.dspy.domain_specific_expert import DomainSpecificExpert as DSPyDomainSpecificExpert
            from ma_finkg.agents.dspy.data_processing_expert import DataProcessingExpert as DSPyDataProcessingExpert
            from ma_finkg.agents.dspy.knowledge_extraction_expert import KnowledgeExtractionExpert as DSPyKnowledgeExtractionExpert
            
            self.kg_expert = DSPyKnowledgeGraphExpert(model_name, prompts=prompts)
            self.domain_specific_expert = DSPyDomainSpecificExpert(ontology=ontology)
            self.data_processing_expert = DSPyDataProcessingExpert()
            
            # Pass optimization parameters to knowledge extraction expert
            optimize = dspy_train_examples is not None and dspy_val_examples is not None
            self.knowledge_extraction_expert = DSPyKnowledgeExtractionExpert(
                model_name, prompts=prompts,
                optimize=optimize, 
                train_examples=dspy_train_examples,
                val_examples=dspy_val_examples
            )
        else:
            # Initialize YAML-based agents (default)
            self.kg_expert = KnowledgeGraphExpert(model_name, prompts=prompts)
            self.domain_specific_expert = DomainSpecificExpert(ontology=ontology)
            self.data_processing_expert = DataProcessingExpert(prompts=prompts)
            self.knowledge_extraction_expert = KnowledgeExtractionExpert(model_name, prompts=prompts, enable_re=enable_re)
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph with all agents and conditional routing logic.
        """
        workflow = StateGraph(GraphState)
        workflow.add_node("kg_expert", self._kg_expert_node)
        workflow.add_node("domain_specific_expert", self._domain_specific_expert_node) 
        workflow.add_node("data_processing_expert", self._data_processing_expert_node)
        workflow.add_node("knowledge_extraction_expert", self._knowledge_extraction_expert_node)
        workflow.add_node("finalize", self._finalize_node)
        
        workflow.set_entry_point("kg_expert")
        
        # Add conditional edges for dynamic routing
        # The KG Expert acts as the central router based on state analysis
        workflow.add_conditional_edges(
            "kg_expert",
            self._route_next_agent,
            {
                "domain_specific_expert": "domain_specific_expert",
                "data_processing_expert": "data_processing_expert", 
                "knowledge_extraction_expert": "knowledge_extraction_expert",
                "finalize": "finalize",
                "END": END
            }
        )
        
        # All other agents return to KG Expert for routing decisions
        workflow.add_edge("domain_specific_expert", "kg_expert")
        workflow.add_edge("data_processing_expert", "kg_expert") 
        workflow.add_edge("knowledge_extraction_expert", "kg_expert")
        workflow.add_edge("finalize", END)
        
        # Compile the graph with checkpointer for state persistence
        return workflow.compile()
    
    def _kg_expert_node(self, state: GraphState) -> Dict[str, Any]:
        """Knowledge Graph Expert node - central coordinator."""
        return self.kg_expert(state)
    
    def _domain_specific_expert_node(self, state: GraphState) -> Dict[str, Any]:
        """Financial Market Expert node - ontology definition.""" 
        return self.domain_specific_expert(state)
    
    def _data_processing_expert_node(self, state: GraphState) -> Dict[str, Any]:
        """Data Processing Expert node - data handling."""
        return self.data_processing_expert(state)
    
    def _knowledge_extraction_expert_node(self, state: GraphState) -> Dict[str, Any]:
        """Knowledge Extraction Expert node - NER and RE."""
        return self.knowledge_extraction_expert(state)
    
    
    def _finalize_node(self, state: GraphState) -> Dict[str, Any]:
        """Finalization node - prepare final output."""
        return self.kg_expert.finalize_kg_construction(state)
    
    def _route_next_agent(self, state: GraphState) -> str:
        """
        Central routing logic that determines which agent to call next.
        """
        next_agent = state.get("next_agent", "data_processing_expert")
        
        # Map next_agent to actual node names
        agent_mapping = {
            "domain_specific_expert": "domain_specific_expert",
            "data_processing_expert": "data_processing_expert", 
            "knowledge_extraction_expert": "knowledge_extraction_expert",
            "finalize": "finalize",
            "END": "END"
        }
        
        return agent_mapping.get(next_agent, "data_processing_expert")
    
    def construct_kg(self, financial_text: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for knowledge graph construction.
        
        Args:
            financial_text: Input text for KG construction
            config: Optional configuration for the graph execution
            
        Returns:
            Final state with constructed knowledge graph
        """
        if config is None:
            config = {"configurable": {"thread_id": "financial_kg_construction"}}
        
        # Handle large texts by chunking
        if len(financial_text) > 5000:
            chunks = chunk_large_text(financial_text)
            print(f"Large text detected ({len(financial_text)} chars), processing {len(chunks)} chunks...")
            return self._process_chunks(chunks, financial_text, config)
        
        # Original processing for normal-sized texts
        return self._process_single_text(financial_text, config)
    
    def _process_single_text(self, financial_text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single text through the original pipeline."""
        # Initialize state with the input text
        initial_state = self.kg_expert.initialize_state(financial_text)
        
        # Start timing
        set_global_timer()
        elapsed = get_elapsed_time()
        print(f"[{elapsed:.1f}s] Starting knowledge graph construction...")
        
        # Execute the graph
        final_state = None
        for state in self.graph.stream(initial_state, config):
            final_state = state
            
            # Print progress with timing
            if "kg_expert" in state:
                next_agent = state["kg_expert"].get("next_agent", "unknown")
                step_names = {
                    "domain_specific_expert": "Creating ontology",
                    "knowledge_extraction_expert": "Extracting entities and relations", 
                    "finalize": "Finalizing results"
                }
                current_step = step_names.get(next_agent, "Processing")
                elapsed = get_elapsed_time()
                print(f"[{elapsed:.1f}s] {current_step}...")
        
        elapsed = get_elapsed_time()
        print(f"[{elapsed:.1f}s] Construction completed!")
        return final_state
    
    def classify_relation(self, sentence: str, e1_text: str, e1_type: str, e2_text: str, e2_type: str) -> str:
        """
        Direct relation classification between specific entity pair (Track B evaluation).
        
        Args:
            sentence: Input sentence containing entities
            e1_text: First entity text 
            e1_type: First entity type
            e2_text: Second entity text
            e2_type: Second entity type
            
        Returns:
            Predicted relation type or "no_relation"
        """
        # Create minimal state for direct classification
        state = {
            "raw_text": sentence,
            "extracted_entities": [
                type("Entity", (), {"text": e1_text, "entity_type": e1_type})(),
                type("Entity", (), {"text": e2_text, "entity_type": e2_type})()
            ],
            "ontology": None
        }
        
        # Get ontology if not available
        if not state["ontology"]:
            ontology_result = self.domain_specific_expert({"raw_text": sentence})
            state["ontology"] = ontology_result.get("ontology")
        
        # Use knowledge extraction expert for direct relation classification
        result = self.knowledge_extraction_expert.classify_entity_pair(
            sentence, e1_text, e1_type, e2_text, e2_type, state["ontology"]
        )
        
        return result if result else "no_relation"
    
    def _process_chunks(self, chunks: list, full_text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process multiple text chunks with shared ontology from full context."""
        # Step 1: Create global ontology from full text context
        print(f"Creating global ontology from full text ({len(full_text)} chars)...")
        temp_state = {"raw_text": full_text}
        ontology_result = self.domain_specific_expert(temp_state)
        global_ontology = ontology_result.get("ontology")
        
        # Step 2: Extract from each chunk using shared ontology
        all_entities = []
        all_triples = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Processing chunk {i}/{len(chunks)} with global ontology...")
            
            # Create chunk state with pre-existing ontology
            chunk_state = {
                "raw_text": chunk,
                "ontology": global_ontology,
                "extracted_entities": [],
                "initial_triples": [],
                "revised_entities": [],
                "revised_triples": [],
                "next_agent": "knowledge_extraction_expert",  # Skip ontology creation
                "messages": []
            }
            
            # Extract using knowledge extraction expert only
            extraction_result = self.knowledge_extraction_expert(chunk_state)
            entities = extraction_result.get("revised_entities", [])
            triples = extraction_result.get("revised_triples", [])
            
            all_entities.extend(entities)
            all_triples.extend(triples)
        
        # Simple deduplication and create merged result
        unique_entities = self._deduplicate_entities(all_entities)
        unique_triples = self._deduplicate_triples(all_triples)
        
        # Return result in same format as single text processing
        return {
            "finalize": {
                "revised_entities": unique_entities,
                "revised_triples": unique_triples,
                "messages": [{"agent": "chunking", "action": "merged_results", 
                            "entities": len(unique_entities), "triples": len(unique_triples)}]
            }
        }
    
    def _deduplicate_entities(self, entities: list) -> list:
        """Remove duplicate entities based on text only."""
        seen = set()
        unique = []
        for entity in entities:
            key = entity.text.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(entity)
        return unique
    
    def _deduplicate_triples(self, triples: list) -> list:
        """Remove duplicate triples based on head, relation, tail."""
        seen = set()
        unique = []
        for triple in triples:
            key = (triple.head.lower().strip(), triple.relation, triple.tail.lower().strip())
            if key not in seen:
                seen.add(key)
                unique.append(triple)
        return unique
    
    def get_final_kg(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts the final knowledge graph from the execution state.
        """
        # Get the last state from the graph execution
        last_agent_state = None
        for agent_key in ["finalize", "kg_expert", "knowledge_extraction_expert"]:
            if agent_key in final_state:
                last_agent_state = final_state[agent_key]
                print(f"[FINAL] Processing results from {agent_key}")
                break
        
        if not last_agent_state:
            return {"error": "No final state found"}
        
        # Extract entities and triples
        entities = last_agent_state.get("revised_entities", [])
        triples = last_agent_state.get("revised_triples", [])
        print(f"[FINAL] Knowledge graph ready: {len(entities)} entities, {len(triples)} triples")
        
        # Create final KG representation
        final_kg = {
            "entities": [
                {
                    "text": entity.text,
                    "type": entity.entity_type
                }
                for entity in entities
            ],
            "triples": [
                {
                    "head": triple.head,
                    "relation": triple.relation,
                    "tail": triple.tail
                }
                for triple in triples
            ],
            "statistics": {
                "total_entities": len(entities),
                "total_triples": len(triples),
                "entity_types": list(set(e.entity_type for e in entities)),
                "relation_types": list(set(t.relation for t in triples)),
                "processing_stats": last_agent_state.get("processing_stats", {})
            },
            "messages": last_agent_state.get("messages", [])
        }
        
        return final_kg
    
    def visualize_graph(self) -> None:
        """
        Visualizes the agent collaboration graph structure.
        """
        try:
            from IPython.display import Image, display
            display(Image(self.graph.get_graph().draw_mermaid_png()))
        except ImportError:
            print("IPython not available. Graph visualization requires Jupyter environment.")
        except Exception as e:
            print(f"Could not visualize graph: {e}")
