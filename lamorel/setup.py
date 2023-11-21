from setuptools import setup, find_packages

setup(
    name='lamorel',
    packages=find_packages("src"),
    package_dir={"": "src"},
    version="0.2",
    install_requires=[
        'transformers>=4.35',
        'accelerate>=0.24.1',
        'hydra-core',
        'torch>=2.1.0'
    ],
    extras_require={
        'quantization':  ["bitsandbytes>=0.41.1"]
    },
    description="",
    author="Clément Romac (Hugging Face & Inria)"
)