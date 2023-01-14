from setuptools import setup, find_packages
from DiabetesComplecationPrediction import __version__

with open("requirements.txt") as f:
    requirements = []
    for library in f.read().splitlines():
        requirements.append(library)

setup(
    author="Caiwei Tan, Ni Yang",
    description="A package that builds prediction models for diabetes complecations",
    name="<DiabetesComplecationPrediction>",
    license="The MIT License (MIT)",
    url="https://github.com/dolores824/hds-package",
    packages=find_packages(include=["<DiabetesComplecationPrediction>", "<DiabetesComplecationPrediction>.*"]),
    install_requires=requirements,
    python_requires='>=3.9',
    version=__version__,
)
