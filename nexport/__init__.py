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


# Dunder attributes
__version__ = "v0.4.7" # update setup.py
__author__ = "Jordan Welsman"

# Import submodules so submodule functions
# are usable at 'import nexport' level
from .calculators import *
from .generic import *
from .models import *
from .utils import *
from .pytorch import __all__
from .tensorflow import __all__

# Initialize super-attribute with framework detection
__framework__ = detect_framework()

# Only show functions specified in
# submodules' __all__ to the outside world
__all__ = calculators.__all__, generic.__all__, models.__all__, utils.__all__, pytorch.__all__, tensorflow.__all__, __framework__
