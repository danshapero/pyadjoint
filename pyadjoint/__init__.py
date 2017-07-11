__version__ = '0.0.1'
__author__  = 'Sebastian Kenji Mitusch'
__credits__ = []
__license__ = 'LGPL-3'
__maintainer__ = 'Sebastian Kenji Mitusch'
__email__ = 'sebastkm@math.uio.no'

from .block import Block
from .tape import Tape, set_working_tape, get_working_tape
from .adjfloat import AdjFloat
from .reduced_functional import ReducedFunctional
from .drivers import compute_gradient, Hessian
from .verification import taylor_test, taylor_test_multiple
