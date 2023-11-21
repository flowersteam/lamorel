from setuptools import setup, find_packages

setup(
    name='lamorel',
    packages=find_packages("src"),
    package_dir={"": "src"},
    version="0.1",
    install_requires=[
        'transformers>=4.35',
        'accelerate>=0.24.1',
        'hydra-core',
        'torch>=2.1.0'
    ],
    description="",
    author=""
)