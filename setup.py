from setuptools import setup, find_packages

setup(
    name="minerva-rag",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'sentence-transformers>=2.2.0',
        'torch>=2.0.0',
        'networkx>=3.0',
        'scikit-learn>=1.0.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
        'ollama>=0.1.0',
        'unstructured>=0.10.0'
    ],
    entry_points={
        'console_scripts': [
            'minerva=cli:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Evolutionary RAG system with holographic memory compression",
    keywords="rag ai machine-learning knowledge-management",
    python_requires=">=3.9",
)
