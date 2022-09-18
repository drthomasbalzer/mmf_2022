################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################


import numpy as np
import plot_utilities as pu

def bivariate_normal_scatter(mu_1, mu_2, sigma_1, sigma_2, rho):

    ### scatter plot of bivariate normal distribution
    ### we sample 2 * sizes from a standard normal distribution and then create a correlated sample set
    ###

    size = 500
    sns = np.random.standard_normal(2 * size)

    x = [sigma_1 * sns[k] + mu_1 for k in range(size)]
    y = [[mu_2 + sigma_2 * (r * sns[k] + np.sqrt(1 - r * r) * sns[k + size]) for k in range(size)] for r in rho]

    colors = ['blue', 'green', 'orange']

    mp = pu.PlotUtilities('Bivariate Normal Distribution with Varying Correlations'.format(rho), 'x', 'y')

    mp.scatterPlot(x, y, rho, colors)


if __name__ == '__main__':

    mu = 0
    sigma_sq = 2.5
#    rho = -0.9
    rho = [-0.9, 0., 0.9]
    bivariate_normal_scatter(mu, mu, sigma_sq, sigma_sq, rho)

