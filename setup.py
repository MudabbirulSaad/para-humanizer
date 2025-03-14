"""
Setup configuration for the UltimateParaphraser package.
"""
from setuptools import setup, find_packages

setup(
    name="para-humanizer",
    version="0.1.0",
    description="Advanced text paraphrasing with human-like qualities",
    author="UltimateParaphraser Team",
    packages=find_packages(),
    install_requires=[
        "nltk>=3.6.0",
        "spacy>=3.2.0",
        "torch>=1.10.0",
        "transformers>=4.12.0",
        "numpy>=1.19.0",
    ],
    entry_points={
        "console_scripts": [
            "paraphraser=para_humanizer.cli.command_line:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.7",
)
