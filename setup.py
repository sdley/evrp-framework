"""
Setup script for RL4EVRP framework.
"""

from setuptools import setup, find_packages

setup(
    name="rl4evrp",
    version="0.1.0",
    description="Deep Reinforcement Learning Framework for Electric Vehicle Routing Problems with Explainable AI",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/rl4evrp",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.23.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.21.0",
    ],
    extras_require={
        "llm": ["groq>=0.4.0"],
        "dev": ["jupyter>=1.0.0", "ipykernel>=6.20.0"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
