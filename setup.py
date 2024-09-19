import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dreamsdm",
    version="0.0.1",
    author="Devina Mohan",
    author_email="TODO",
    description="DREAMS DM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/devinamhn/dreams-dm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
)