from pathlib import Path
from typing import Dict, Any

# Project root directory
PROJECT_ROOT = Path(__file__).parent

class Config:    
    # Core directories
    ONTOLOGY_DIR = PROJECT_ROOT / "ontology"
    PROMPTS_DIR = PROJECT_ROOT / "prompts"  
    UTILS_DIR = PROJECT_ROOT / "utils"
    AGENTS_DIR = PROJECT_ROOT / "agents"
    EVALS_DIR = PROJECT_ROOT / "evals"
    
    # Key files
    ONTOLOGY_FIRE = ONTOLOGY_DIR / "ontology_fire.json"
    ONTOLOGY_REFIND = ONTOLOGY_DIR / "ontology_refind.json"
    ONTOLOGY_NYT11 = ONTOLOGY_DIR / "ontology_nyt11.json"
    ONTOLOGY_CONLLPP = ONTOLOGY_DIR / "ontology_conllpp.json"
    ONTOLOGY_DEFAULT = ONTOLOGY_DIR / "ontology_default.json"
    
    # Prompt files
    PROMPTS_FIRE = PROMPTS_DIR / "prompts_fire.yaml"
    PROMPTS_REFIND = PROMPTS_DIR / "prompts_refind.yaml"
    PROMPTS_NYT11 = PROMPTS_DIR / "prompts_nyt11.yaml"
    PROMPTS_CONLLPP = PROMPTS_DIR / "prompts_conllpp.yaml"
    PROMPTS_DEFAULT = PROMPTS_DIR / "prompts_default.yaml"
    
    # Output directories 
    OUTPUT_DIR = PROJECT_ROOT / "output"
    EVAL_OUTPUT_DIR = EVALS_DIR / "output"
    
    @classmethod
    def ensure_output_dirs(cls):
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.EVAL_OUTPUT_DIR.mkdir(exist_ok=True)
    
    @classmethod 
    def get_prompts_file(cls, prompt: str = "default") -> Path:
        prompts_map = {
            "fire": cls.PROMPTS_FIRE,
            "refind": cls.PROMPTS_REFIND,
            "nyt11": cls.PROMPTS_NYT11, 
            "conllpp": cls.PROMPTS_CONLLPP,
            "default": cls.PROMPTS_DEFAULT
        }
        if prompt.lower() not in prompts_map and prompt != "default":
            print(f"[CONFIG] Prompt '{prompt}' not found, using default prompts")
        return prompts_map.get(prompt.lower(), cls.PROMPTS_DEFAULT)
    
    @classmethod 
    def get_ontology_file(cls, ontology: str = "default") -> Path:
        ontology_map = {
            "fire": cls.ONTOLOGY_FIRE,
            "refind": cls.ONTOLOGY_REFIND,
            "nyt11": cls.ONTOLOGY_NYT11,
            "conllpp": cls.ONTOLOGY_NYT11,
            "default": cls.ONTOLOGY_DEFAULT
        }
        if ontology.lower() not in ontology_map and ontology != "default":
            print(f"[CONFIG] Ontology '{ontology}' not found, using default ontology")
        return ontology_map.get(ontology.lower(), cls.ONTOLOGY_DEFAULT)