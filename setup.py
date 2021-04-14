# Copyright 2018 Zhao Xingyu & An Yuexuan. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

with open('README.rst', 'r', encoding='utf8') as f:
    long_description = f.read()
setup(
    name='ailearn',
    version='0.2.1.3',
    description='A lightweight package for artificial intelligence',
    long_description=long_description,
    author='ZHAO Xingyu; AN Yuexuan',
    author_email='757008724@qq.com',
    license='Apache License, Version 2.0',
    url='http://github.com/axi345/ailearn',
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Environment :: Console',
    ],
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'sklearn', 'matplotlib', 'scipy', 'keras', 'bs4', 'requests'],
)
