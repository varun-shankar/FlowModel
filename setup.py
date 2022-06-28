from setuptools import setup, find_packages
from pathlib import Path

# see https://packaging.python.org/guides/single-sourcing-package-version/
# version_dict = {}
# with open(Path(__file__).parents[0] / "nequip/_version.py") as fp:
#     exec(fp.read(), version_dict)
# version = version_dict["__version__"]
# del version_dict

setup(
    name="flowmodel",
    description="Equivariant GNN for flow modeling",
    author="Varun Shankar",
    python_requires=">=3.6",
    packages=find_packages(include=["flowmodel", "flowmodel.*"]),
    install_requires=[
        "numpy",
        "torch>=1.8",
        "e3nn>=0.3.3",
        "pyyaml",
        "packaging",
        "pytorch-lightning",
        "setuptools>=59.5.0",
        "pytorch-lightning",
        "wandb"
    ],
    zip_safe=True,
)