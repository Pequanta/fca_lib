from setuptools import setup, find_packages

def parse_requirements():
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="fca-project",
    version="0.1.0",
    description="A python library for Formal Concept Analysis (FCA) and related algorithms including QUBO formulations.",
    packages=find_packages(include=["fca*", "cli*", "scripts*"]),
    install_requires=parse_requirements(),
)
