"""Setup script for RL4EVRP."""

from setuptools import find_packages, setup

setup(
    name="rl4evrp",
    version="0.1.0",
    description="Deep reinforcement learning framework for electric vehicle routing",
    author="RL4EVRP Contributors",
    python_requires=">=3.9",
    packages=find_packages(exclude=("tests", "scripts")),
    install_requires=[
        "torch==2.0.1",
        "numpy>=1.23.0,<2.0",
        "pandas>=1.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.21.0",
        "groq>=0.4.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.20.0",
            "pytest>=8.0.0",
        ]
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
