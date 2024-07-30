# COPYRIGHT NOTICE

# “Neural Network Export Package (nexport) v0.4.6” Copyright (c) 2023,
# The Regents of the University of California, through Lawrence Berkeley
# National Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.

# If you have questions about your rights to use or distribute this software,
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

# NOTICE. This Software was developed under funding from the U.S. Department
# of Energy and the U.S. Government consequently retains certain rights. As
# such, the U.S. Government has been granted for itself and others acting on
# its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
# Software to reproduce, distribute copies to the public, prepare derivative
# works, and perform publicly and display publicly, and to permit others to do so.


# Module imports
from setuptools import setup

# Arguments
version = "0.4.7" # update __init__.py
python_version = ">=3.10"

# Long description from README.md
with open("README.md", "r") as fh:
    long_description = fh.read()

# Define list of submodules
py_modules = ["calculators", "colors", "generic", "models", "utils", "pytorch", "tensorflow"]

# Run stup function
setup(
    name = 'nexport',
    version = version,
    description = 'A Python package for exporting the weights and biases of neural networks.',
    license='LBNL BSD',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = 'Jordan Welsman',
    author_email = 'welsman@lbl.gov',
    url = 'https://pypi.org/project/nexport/',
    download_url='https://github.com/JordanWelsman/nexport/tags',
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
    ],
    package_data = {
      'nexport': py_modules
      },
    python_requires=python_version,
    install_requires = [
        "jutl",
        "numpy<1.24",
    ],
    keywords='python, neural network, export, import, parameters, weights, biases, layers, neurons'
)
