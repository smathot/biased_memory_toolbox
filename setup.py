#!/usr/bin/env python
# coding=utf-8
"""
This file is part of biased memory toolbox.

biased memory toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

biased memory toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with biased memory toolbox.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import biased_memory_toolbox as bmt
from setuptools import setup


if os.path.exists('readme.md'):
    with open('readme.md') as fd:
        readme = fd.read()
else:
    readme = ''

setup(
    name='biased_memory_toolbox',
    version=bmt.__version__,
    description='Mixture modeling for working-memory experiments',
    author=u'Sebastiaan Mathot',
    author_email=u's.mathot@cogsci.nl',
    long_description=readme,
    long_description_content_type='text/markdown',
    license=u'GNU GPL Version 3',
    py_modules=['biased_memory_toolbox'],
    url=u'https://github.com/smathot/biased_memory_toolbox',
    install_requires=['scipy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ]
)
