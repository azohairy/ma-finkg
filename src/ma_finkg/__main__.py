#!/usr/bin/env python3
"""Command-line interface for ma_finkg package."""

import sys
import argparse

def show_help():
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
    print(f"Found {len(result.entities)} entities and {len(result.triples)} relations")
    print(result.entities)
    print(result.triples)

PARAMETERS:
    model: "openai/gpt-4o-mini" (default), "openai/gpt-3.5-turbo"
    ontology: Choose from available ontologies
    prompts: Choose from available prompt sets
    use_dspy: False (default), True for optimized extraction

AVAILABLE ONTOLOGIES:
    • "fire" (default) - FIRE financial dataset with financial entities & relations
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
    • "default" - Generic financial ontology

AVAILABLE PROMPT SETS:
    • "fire" (default) - Optimized for FIRE financial dataset
    • "nyt11" - Optimized for NYT dataset  
    • "conllpp" - Optimized for CoNLL++ dataset
    • "refind" - Optimized for ReFiNED dataset
    • "default" - Generic prompts for financial text

EXAMPLES:
    # Use FIRE ontology (best for financial text)
    kg = MA_FinKG(ontology="fire", prompts="fire")
    
    # Use NYT ontology (general purpose)
    kg = MA_FinKG(ontology="nyt11", prompts="nyt11")
    
    # Command line usage
    python -m ma_finkg --help
    python -m ma_finkg --version
    ma-finkg --help
    """.strip())

def show_version():
    """Display version information."""
    from ma_finkg import __version__
    print(f"ma-finkg {__version__}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Financial Knowledge Graph Construction",
        add_help=False
    )
    parser.add_argument("--help", action="store_true", help="Show help message")
    parser.add_argument("--version", action="store_true", help="Show version")
    
    args = parser.parse_args()
    
    if args.help:
        show_help()
    elif args.version:
        show_version()
    else:
        show_help()

if __name__ == "__main__":
    main()