#!/usr/bin/env python
"""Setup script for Calabi-Yau ML package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=1.21.0,<2.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "torch>=2.0.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.14.0",
    "shap>=0.42.0,<0.50.0",
    "tqdm>=4.65.0",
]

# Optional dependencies for advanced features
EXTRAS_REQUIRE = {
    "gnn": [
        "torch-geometric>=2.3.0",
        "torch-scatter>=2.1.0",
        "torch-sparse>=0.6.17",
    ],
    "bayesian": [
        "pyro-ppl>=1.8.0",
    ],
    "symbolic": [
        "pysr>=0.15.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-xdist>=3.0.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "isort>=5.12.0",
        "pre-commit>=3.0.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "sphinx-autodoc-typehints>=1.22.0",
    ],
}

# All optional dependencies
EXTRAS_REQUIRE["all"] = sum(EXTRAS_REQUIRE.values(), [])

setup(
    name="calabi-yau-ml",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep Learning for Calabi-Yau Manifolds: Geometry to Physics Mapping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/calabi-yau-ml",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks", "notebooks.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "cy-train=calabi_yau.cli.train:main",
            "cy-evaluate=calabi_yau.cli.evaluate:main",
            "cy-generate=calabi_yau.cli.generate:main",
        ],
    },
    include_package_data=True,
    package_data={
        "calabi_yau": ["configs/*.yaml", "data/*.txt"],
    },
    zip_safe=False,
)
