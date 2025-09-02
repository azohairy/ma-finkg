# MA-FINKG: Multi-Agent Framework for Financial Knowledge Graph Construction from Unstructured Text

A multi-agent system based on LangGraph for constructing Knowledge Graphs from financial text documents. The system uses specialized agents to coordinate domain-specific ontology building, text processing, and knowledge extraction.

## Features

- **Multi-Agent Architecture**: Coordinated agents for different aspects of KG construction
- **Benchmarked on varity of datasets**: Support for FIRE, NYT11, CoNLLpp, REFinD, and custom ontologies
- **Flexible Prompting**: Specialized prompt sets optimized for different datasets
- **DSPy Integration**: Optional optimization for improved extraction performance
- **Financial Focus**: Specialized for financial text analysis and knowledge extraction

## Installation

```bash
pip install ma-finkg
```

Or install from source:

```bash
git clone <repository-url>
cd financial_kg_system
pip install -e .
```

## Quick Start

```python
from ma_finkg import MA_FinKG

# Initialize with API key
kg = MA_FinKG(api_key="your-openrouter-key")

# Or set environment variable
import os
os.environ["OPENROUTER_API_KEY"] = "your-key"
kg = MA_FinKG()

# Generate knowledge graph
text = "Apple Inc. reported revenue of $95B in Q4 2023."
result = kg.generate(text)

# Access results
print(f"Found {len(result.entities)} entities and {len(result.triples)} relations")
print("Entities:", result.entities)
print("Relations:", result.triples)
```

## Available Ontologies

### FIRE
Financial dataset ontology with comprehensive financial entities and relations.

**Entities**: Action, BusinessUnit, Company, Date, Designation, FinancialEntity, GeopoliticalEntity, Location, Money, Person, Product, Quantity, Sector

**Relations**: ActionBuy, ActionSell, ActionMerge, Employeeof, Subsidiaryof, Value, Locatedin, etc. (18 total)

### NYT11
General-purpose ontology based on the NYT dataset.

**Entities**: PER (Person), ORG (Organization), LOC (Location)

**Relations**: /people/person/place_of_birth, /business/company/founders, /location/country/capital, etc. (12 total)

### Default (Default)
Generic financial ontology used when no specific dataset is specified.

### Other Ontologies
- **CoNLLpp**: Named Entity Recognition dataset ontology
- **REFinD**: ReFiNED dataset ontology

## Configuration Options

```python
kg = MA_FinKG(
    model="openai/gpt-4o-mini",  # or "google/gemini-2.5-flash-lite"
    ontology="default",          # "default" (default), "fire", "nyt11", "conllpp", "refind"
    prompts="default",           # "default" (default), matching prompt set
    use_dspy=False,              # True for optimized extraction
    api_key="your-key"           # OpenRouter API key
)
```

## Multi-Agent Architecture

The system employs four specialized agents:

1. **KG Expert**: Task coordinator that manages the overall KG construction process
2. **Domain Specific Expert**: Builds domain-specific ontologies from input text
3. **Data Processing Expert**: Performs lossless text cleaning and preparation
4. **Knowledge Extraction Expert**: Orchestrates NER and RE sub-agents for entity and relation extraction

## Command Line Usage

```bash
# Show help
ma-finkg --help
python -m ma_finkg --help

# Show version
ma-finkg --version
python -m ma_finkg --version
```


## Requirements

- Python 3.9+
- langchain-core
- langchain-openai
- langgraph
- dspy-ai
- pydantic
- pyyaml
- json-repair

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{ma-finkg,
  title={MA-FinKG: Multi-Agent System for Financial Knowledge Graph Construction for Text},
  author={Abdel-Rahman Elzohairy},
  version={0.1.5},
  year={2025}
}
```