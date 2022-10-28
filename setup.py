# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

extras = {
    "datagen": [
        "phiflow",
        "joblib",
        "juliapkg",
        "tqdm",
    ],
}

base_requires = [
    "pytorch-lightning>=1.7",
    "numpy",
    "xarray",
    "h5py",
    "click",
    "torch>=1.12",
    "torchdata",
    "matplotlib",
    "jsonargparse",
    "omegaconf",
    "tensorboard"
]

setup(
    name="pdearena",
    description="PyTorch library for PDE Surrogate Benchmarking",
    install_requires=base_requires,
    extras_require=extras,
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    version="0.1.0",
    author="Jayesh K. Gupta <jayesh.gupta@microsoft.com>, Johannes Brandstetter <johannesb@microsoft.com>",
    python_requires=">=3.6",
    zip_safe=True,
    project_urls={
        "Documentation": "https://microsoft.github.io/pdearena",
        "Source code": "https://github.com/microsoft/pdearena",
        "Bug Tracker": "https://github.com/microsoft/pdearena/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
