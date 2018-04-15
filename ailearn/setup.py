# -*- coding: utf-8 -*-
from setuptools import setup

PACKAGE = "ailearn"
NAME = "ailearn"
DESCRIPTION = "A lightweight package for artificial intelligence"
AUTHOR = "ZHAO Xingyu; AN Yuexuan"
AUTHOR_EMAIL = "757008724@qq.com"
URL = "http://github.com/axi345/ailearn"
VERSION = __import__(PACKAGE).__version__

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    # long_description=read("README.rst"),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license="BSD",
    url=URL,
    packages=[PACKAGE],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    zip_safe=False,
)
