"""
Multi-Agent Framework for Financial Knowledge Graph Construction using LangGraph

This package implements a multi-agent system based on LangGraph for constructing
Knowledge Graphs from financial text documents.
"""

__version__ = "0.1.7"
__author__ = "Abdel-Rahman Elzohairy"

import os

# Import main classes
from .kg_construction_graph import FinancialKGConstructionGraph

class KnowledgeGraph:
    """Unified knowledge graph representation."""
    def __init__(self, entities, triples, statistics):
        self.entities = entities
        self.triples = triples  
        self.statistics = statistics
    
    def __len__(self):
        return len(self.entities) + len(self.triples)
    
    def __repr__(self):
        return f"KnowledgeGraph(entities={len(self.entities)}, triples={len(self.triples)})"

class MA_FinKG:
    """Simplified interface for financial knowledge graph construction."""
    
    def __init__(self, model="openai/gpt-4o-mini", api_key=None, ontology="default", prompts="default", use_dspy=False):
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
        
        self._kg_system = FinancialKGConstructionGraph(
            model_name=model,
            ontology=ontology, 
            prompts=prompts,
            use_dspy=use_dspy
        )
    
    def generate(self, input):
        """Generate knowledge graph from input text."""
        final_state = self._kg_system.construct_kg(input)
        final_kg = self._kg_system.get_final_kg(final_state)
        
        return KnowledgeGraph(
            entities=final_kg.get("entities", []),
            triples=final_kg.get("triples", []),
            statistics=final_kg.get("statistics", {})
        )

def help():
    """Display usage information."""
    print("""
MA-FINKG: Multi-Agent Financial Knowledge Graph Construction

USAGE:
    from ma_finkg import MA_FinKG
    
    # Initialize (requires OPENROUTER_API_KEY environment variable)
    kg = MA_FinKG()
    
    # Or pass API key directly
    kg = MA_FinKG(api_key="your-openrouter-key")
    
    # Generate knowledge graph
    result = kg.generate("Apple Inc. reported revenue of $95B in Q4 2023.")
    
    # Access results
    print(f"Found {len(result.entities)} entities and {len(result.triples)} relations")
    print(result.entities)  # List of entities
    print(result.triples)   # List of relations

PARAMETERS:
    model: "openai/gpt-4o-mini" (default), "openai/gpt-3.5-turbo"
    ontology: Choose from available ontologies
    prompts: Choose from available prompt sets  
    use_dspy: False (default), True for optimized extraction

AVAILABLE ONTOLOGIES:
    • "fire" - FIRE financial dataset with financial entities & relations
      Entities: Action, BusinessUnit, Company, Date, Designation, FinancialEntity, 
                GeopoliticalEntity, Location, Money, Person, Product, Quantity, Sector
      Relations: ActionBuy, ActionSell, ActionMerge, Employeeof, Subsidiaryof, 
                 Value, Locatedin, etc. (18 total)
                 
    • "nyt11" - NYT dataset with general entities & relations
      Entities: PER (Person), ORG (Organization), LOC (Location)  
      Relations: /people/person/place_of_birth, /business/company/founders, 
                 /location/country/capital, etc. (12 total)
                 
    • "conllpp" - CoNLL++ NER dataset
    • "refind" - ReFiNED dataset  
    • "default" (default) - Generic financial ontology

AVAILABLE PROMPT SETS:
    • "fire" - Optimized for FIRE financial dataset
    • "nyt11" - Optimized for NYT dataset
    • "conllpp" - Optimized for CoNLL++ dataset
    • "refind" - Optimized for ReFiNED dataset
    • "default" (default) - Generic prompts for financial text

EXAMPLES:
    # Use default ontology (generic financial)
    kg = MA_FinKG()  # Uses default ontology and prompts
    
    # Use FIRE ontology (best for financial text)
    kg = MA_FinKG(ontology="fire", prompts="fire")
    
    # Use NYT ontology (general purpose) 
    kg = MA_FinKG(ontology="nyt11", prompts="nyt11")
    """)

__all__ = ["MA_FinKG", "KnowledgeGraph"]