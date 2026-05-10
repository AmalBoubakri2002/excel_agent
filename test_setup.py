def test_imports():
    import pandas as pd
    import numpy as np
    import sklearn
    import langchain
    import langgraph
    import pydantic
    print(" Toutes les librairies sont importées avec succès")

def test_ollama():
    from langchain_ollama import OllamaLLM
    
    llm = OllamaLLM(model="llama3.2:3b")
    response = llm.invoke("Réponds juste 'OK' sans rien d'autre.")
    print(f" Ollama répond : {response}")

if __name__ == "__main__":
    test_imports()
    test_ollama()