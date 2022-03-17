import sys
import os
from pathlib import Path
from setuptools import setup, find_packages
from glob import glob

if sys.version_info < (3, 6):
    sys.exit("This software requires Python >= 3.6")


def get_requirements(reqs_file: str):
    return [l.strip() for l in Path(reqs_file).read_text("utf-8").splitlines()]


setup(
    name="MyProject",
    version="0.1",
    description="MyProject description",
    long_description=Path("README.md").read_text("utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/calico/MyProject",
    author="MyProjectAuthor",
    author_email="MyProjectAuthor@calicolabs.com",
    python_requires=">=3.6",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[
        os.path.splitext(os.path.basename(path))[0] for path in glob("src/*.py")
    ],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
    ],
)
