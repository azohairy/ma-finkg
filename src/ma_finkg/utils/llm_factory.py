import os
import time
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from langchain_openai import ChatOpenAI


def timeout_llm_call(llm, messages, timeout_seconds=40):
    """Surgical timeout wrapper for any LLM call"""
    def _call():
        result = llm.invoke(messages)
        return result
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_call)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError:
            print(f"HUNG: LLM call exceeded {timeout_seconds}s")
            raise Exception(f"LLM call timeout after {timeout_seconds}s")


def create_openrouter_llm(model_name: str = "openai/gpt-3.5-turbo") -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for OpenRouter with standard settings."""
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        request_timeout=30,  
        max_retries=0  
    )


def create_dspy_llm(model_name: str = "openai/gpt-3.5-turbo"):
    """Create a DSPy-compatible LM instance for DSPy agents."""
    # Fix OpenAI/LiteLLM compatibility: create aliases for renamed classes
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
    
    dspy_llm = dspy.LM(
        model=model_name,
        api_base="https://openrouter.ai/api/v1", 
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
        max_tokens=1000
    )
    
    dspy.settings.configure(lm=dspy_llm)
    
    return dspy_llm