"""
Setup configuration for the Para-Humanizer package.
"""
import os
import sys
from setuptools import setup, find_packages

# Dynamically determine the version from environment or default
version = os.environ.get("PARA_HUMANIZER_VERSION", "0.1.0")

# Get long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get dependencies based on environment
install_requires = [
    "nltk>=3.6.0",
    "spacy>=3.2.0",
    "torch>=1.10.0",
    "transformers>=4.12.0",
    "numpy>=1.19.0",
    "fastapi>=0.70.0",
    "uvicorn>=0.15.0",
    "pydantic>=1.9.0",
    "python-dotenv>=0.19.0",
    "pyperclip>=1.8.0",
]

# Add Gradio for the web interface
install_requires.append("gradio>=3.0.0")

# Optional dependencies
extras_require = {
    "dev": [
        "pytest>=6.0.0",
        "black>=21.5b2",
        "isort>=5.9.0",
        "mypy>=0.910",
        "flake8>=3.9.0",
    ],
    "docs": [
        "sphinx>=4.0.0",
        "sphinx-rtd-theme>=1.0.0",
    ],
}

setup(
    name="para-humanizer",
    version=version,
    description="Advanced text paraphrasing with human-like qualities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Para-Humanizer Team",
    author_email="mudabbirulsaad@gmail.com",
    url="https://github.com/mudabbirulsaad/para-humanizer",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
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
