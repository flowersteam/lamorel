from setuptools import setup, find_packages

setup(
    name='lamorel',
    packages=find_packages("src"),
    package_dir={"": "src"},
    version="0.3",
    install_requires=[
        'transformers>=4.35',
        'accelerate>=0.24.1',
        'hydra-core',
        'torch>=2.1.0'
    ],
    extras_require={
        'unsloth':  ["unsloth"]
    },
    description="Lamorel is a Python library designed for people eager to use Large Language Models (LLMs) in interactive environments (e.g. RL setups).",
    author="Clément Romac"
)