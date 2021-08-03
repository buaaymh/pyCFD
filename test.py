import numpy as np

from fdm import FirstOrderScalarScheme
from fdm import WenoScalarScheme



if __name__ == '__main__':

    # Set Scheme:
    solver = FirstOrderScalarScheme()
    # solver = WenoScalarScheme()