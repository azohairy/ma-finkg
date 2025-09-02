from typing import TypedDict, List, Dict, Optional, Any
from pydantic import BaseModel

class EntityType(BaseModel):
    name: str
    examples: List[str]

class RelationType(BaseModel):
    name: str
    head_types: List[str]
    tail_types: List[str]

class FinancialOntology(BaseModel):
    entity_types: Dict[str, EntityType]
    relation_types: Dict[str, RelationType]

class Entity(BaseModel):
    text: str
    entity_type: str

class Triple(BaseModel):
    head: str
    relation: str
    tail: str

class ValidationError(BaseModel):
    error_type: str
    description: str

class GraphState(TypedDict):
    raw_text: str
    ontology: Optional[FinancialOntology]
    extracted_entities: List[Entity]
    initial_triples: List[Triple]
    revised_entities: List[Entity]
    revised_triples: List[Triple]
    next_agent: str
    messages: List[Dict[str, Any]]