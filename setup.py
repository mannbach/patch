"""Script to install local packages
"""
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="patch",
    version="0.0.1",
    author="Jan Bachmann",
    author_email="jan@mannbach.de",
    description=("A model to study the emergence of network inequalities based on "
        "[P]referential [A]ttachment [T]riadic [C]losure and [H]omophily."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mannbach/patch",
    project_urls={
        "Bug Tracker": "https://github.com/mannbach/patch/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="patch"),
    python_requires=">=3.9",
)
