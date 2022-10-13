from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

extras = {
    "datagen": ["phiflow", "joblib"]
}

setup(
    name="pdearena",
    description="PyTorch library for PDE Surrogate Benchmarking",
    install_requires=[],
    extras_require=extras,
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    version="0.1.0",
    author="Jayesh K. Gupta <jayesh.gupta@microsoft.com>, Johannes Brandstetter <johannesb@microsoft.com>",
    python_requires=">=3.6",
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
    ],    
)
