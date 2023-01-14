# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="Caiwei Tan, Ni Yang",
    description="A package that builds prediction models for diabetes complecations",
    name="<DiabetesComplecationPrediction>",
    packages=find_packages(include=["<DiabetesComplecationPrediction>", "<DiabetesComplecationPrediction>.*"]),
    
    #install_requires=[
    #    'pandas',            # any versions of pandas
    #    'matplotlib>=2.2.1'
    #],
    #python_requires='>=3.0',
    
    version="0.1.5",
)
# Run this to install the package locally in editable mode
# pip install -e .
