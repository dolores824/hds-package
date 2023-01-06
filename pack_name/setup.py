# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="<your-name>",
    description="<pack desc>",
    name="<pack_name>",
    packages=find_packages(include=["<pack_name>", "<pack_name>.*"]),
    version="0.1.0",
)
# Run this to install the package locally in editable mode
# pip install -e .
