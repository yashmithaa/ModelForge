from setuptools import setup, find_packages

setup(
    name="modelforge",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "rich",
        "pandas",
        "numpy",
        "nltk",
        "torch",
        "transformers",
        "scikit-learn",
        "pyyaml",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "modelforge=src.cli:main", 
        ],
    },
    include_package_data=True,
    author="Team-ModelForge",
    description="A declarative framework for building machine learning models",
)
