from setuptools import setup, find_packages

import os
import sys

version_file_path = os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"nlutestframework",
	"version.py"
)

version = {}
with open(version_file_path) as f:
	exec(f.read(), version)
version = version["__version__"]

with open("README.md") as f:
    long_description = f.read()

setup(
    name = "NLUTestFramework",
    version = version,
    description = "A framework to benchmark and compare NLU frameworks.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/emundo/nlutestframework",
    author = "Tim Henkes",
    author_email = "emubot@e-mundo.de",
    license = "Apache 2.0",
    packages = find_packages(),
    entry_points = {
        "console_scripts": [
            "nlutestframework=nlutestframework.__main__:main"
        ],
    },
    install_requires = [
        "snips-nlu>=0.20.0,<0.21",
        "dialogflow>=0.7.2,<0.8",
        "docker>=4.1.0,<5",
        "requests>=2.22,<3",
        "matplotlib>=3.1.2,<4",
        "pyyaml>=5.1.2,<6",
        "azure-cognitiveservices-language-luis>=0.5.0,<0.6",
        "langcodes>=1.4.1,<2"
    ],
    python_requires = ">=3.7, <4",
    zip_safe = False,
    classifiers = [
        "Development Status :: 4 - Beta",

        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",

        "Topic :: Communications :: Chat",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",

        "License :: OSI Approved :: Apache Software License",

        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ]
)
