from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sago",
    version="0.1.0",
    author="Zeguan Xiao",
    author_email="hainanxzg@gmail.com",
    description="Sign-Align Gradient Optimization for LLM Unlearning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sustech-nlp/SAGO",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,  # Uses requirements.txt
    extras_require={
        "lm-eval": [
            "lm-eval==0.4.8",
        ],  # Install using `pip install .[lm-eval]`
        "dev": [
            "pre-commit==4.0.1",
            "ruff==0.6.9",
        ],  # Install using `pip install .[dev]`
    },
    python_requires=">=3.11",
)
