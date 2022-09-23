################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################



import numpy as np

import core_math_utilities as dist
import plot_utilities as pu

def comparison_poi_binom(lam, upper_bound):

    n = upper_bound + 1
    x_ax = [k for k in range(n)]
    n_plots = 2  # plotting both poisson and binomial distribution
    y_ax = [0.] * n_plots
    y_ax[0] = [dist.poisson_pdf(lam, k) for k in range(n)]  # poisson distribution
    y_ax[1] = [dist.binomial_pdf(lam / n, k, n) for k in range(n)]

    mp = pu.PlotUtilities("Poisson Vs Binomial distribution for lambda={0}".format(lam), "# Successes", "Probability")
    mp.multiPlot(x_ax, y_ax, '*')


def poisson_plot(lam, upper_bound):

    n = upper_bound + 1
    x_ax = [k for k in range(n)]
    y_ax = [dist.poisson_pdf(lam, k) for k in range(n)]  # poisson distribution

    mp = pu.PlotUtilities("Poisson Distribution for lambda={0}".format(lam), "# Successes", "Probability")
    mp.multiPlot(x_ax, [y_ax], 'o')

def binomial_plot(p, n):

    x_ax = [k for k in range(n)]
    y_ax = [dist.binomial_pdf(p, k, n) for k in range(n)]  # poisson distribution

    mp = pu.PlotUtilities("Binomial Distribution for p={0}".format(p), "# Successes", "Probability")
    mp.multiPlot(x_ax, [y_ax], 'o')



if __name__ == '__main__':

    binomial_plot(0.2, 50)
    poisson_plot(0.5, 10)
    comparison_poi_binom(0.5,10)


