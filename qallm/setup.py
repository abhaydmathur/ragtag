import codecs
import os

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "numpy>=1.24.1",
    "pandas>=1.5.2",
    "setuptools",
    "lingua-language-detector",
]


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r", encoding="utf-8") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="qallm",
    version=get_version("qallm/__init__.py"),
    description="TO BE ADDED.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Paul Houssel",
    author_email="paul.houssel@ip-paris.fr",
    license="MIT",
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    entry_points="""
        [console_scripts]
        qallm=qallm.main:main
    """,
)
