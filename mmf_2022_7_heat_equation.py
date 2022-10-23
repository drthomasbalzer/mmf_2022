################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

import core_math_utilities as dist


def transition_density(_x, _y, _t):

    return dist.normal_pdf(_x, _y, _t)


def expected_positive_exposure(_mean, _variance):
    y = _mean / np.sqrt(_variance)
    return _mean * dist.standard_normal_cdf(y) + np.sqrt(_variance) * dist.standard_normal_pdf(y)


def excess_probability_payoff(_strike, _mean, _variance):
    return 1 - dist.standard_normal_cdf((_strike - _mean) / np.sqrt(_variance))


def second_moment(_mean, _variance):
    return _mean * _mean + _variance


def plot_transition_probability(calculationOption):

    # set up the axes for the first plot
    z_lim = 1.0

    ## preparation of charts
    if (calculationOption == 0 or calculationOption == 1):
        X = np.arange(0.1, 3, 0.2)
        Y = np.arange(-3, 3, 0.1)
    elif (calculationOption == 2 or calculationOption == 3):
        X = np.arange(0.05, 3, 0.1)
        Y = np.arange(-2, 2, 0.1)

    X, Y = np.meshgrid(X, Y)

    ### set up x, y values and populate function values
    ## calculation options
    ## 0 - transition probability
    ## 1 - second moment
    ## 2 - excess probability (digital option payout)
    ## 3 - expected positive exposure

    if (calculationOption == 0):
        Z = transition_density(0, Y, X)
    elif (calculationOption == 1):
        Z = second_moment(Y, X)
        z_lim = 10
    elif (calculationOption == 2):
        Z = excess_probability_payoff(-0.5, Y, X)
    elif (calculationOption == 3):
        Z = expected_positive_exposure(Y, X)

    ### set up proportions of the figure
    fig = plt.figure(figsize=plt.figaspect(0.6))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_zlim(0, z_lim)
    fig.colorbar(surf, shrink=0.75, aspect=8)

    plt.show()


if __name__ == '__main__':

    for k in range(4):
        plot_transition_probability(k)


