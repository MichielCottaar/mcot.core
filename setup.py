#!/usr/bin/env python

"""The setup script."""

import codecs
import os
import re

from setuptools import setup, find_packages

###################################################################

PACKAGES = []
META_PATH = os.path.join("mcot", "core", "__init__.py")

KEYWORDS = ['mcot.core', ]
CLASSIFIERS = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

###################################################################

about = {}
with open(os.path.join("mcot", "core", "__init__.py")) as f:
    exec(f.read(), about)

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

setup(
    name=about["__title__"],
    author=about["__author__"],
    author_email=about["__email__"],
    version=about["__version__"],
    description=about["__summary__"],
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/x-rst",
    license=about["__license__"],
    url=about["__url__"],
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    install_requires=requirements,
    include_package_data=True,
    packages=find_packages(include=['mcot.core']),
    zip_safe=False,
)
