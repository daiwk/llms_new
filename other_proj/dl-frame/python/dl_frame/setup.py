#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='dl-frame',
    version="0.0.0.8",
    description=(
        'desc'
    ),
#    long_description=open('README.rst').read(),
    author="daiwk",
    author_email="daiwk@foxmail.com",
    maintainer="daiwk",
    maintainer_email="daiwk@foxmail.com",
    license='BSD License',
    packages=find_packages(),
    platforms=["all"],
    url='https://github.com/daiwk/dl-frame',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
#        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
#        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries'
    ],
)
