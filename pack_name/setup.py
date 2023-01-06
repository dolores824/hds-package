# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="<your-name>",
    description="<pack desc>",
    name="<pack_name>",
    packages=find_packages(include=["<pack_name>", "<pack_name>.*"]),
    
    #install_requires=[
    #    'pandas',            # any versions of pandas
    #    'matplotlib>=2.2.1'
    #],
    #python_requires='>=3.0',
    
    version="0.1.0",
)
# Run this to install the package locally in editable mode
# pip install -e .
