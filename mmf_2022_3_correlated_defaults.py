################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################

import numpy as np
import matplotlib.pyplot as plt

import core_math_utilities as dist
import plot_utilities as pu


###########
##### Demo of Correlated Default Times

def getCorrelatedStandardNormals(size, rho):

    sns = np.random.standard_normal(2 * size)

    x = [sns[k] for k in range(size)]
    y = [rho * sns[k] + np.sqrt(1 - rho * rho) * sns[k + size] for k in range(size)]

    return x, y

def correlated_defaults_scatter(lambda_1, lambda_2, rhos, size):

    tau_2 = [0] * len(rhos)

    sns = np.random.standard_normal(2 * size)

    x = [sns[k] for k in range(size)]
    tau_1 = [dist.exponential_inverse_cdf(lambda_1, dist.standard_normal_cdf(x1)) for x1 in x]

    index = 0
    for rho in rhos:
        y = [rho * sns[k] + np.sqrt(1 - rho * rho) * sns[k + size] for k in range(size)]
        tau_2[index] = [dist.exponential_inverse_cdf(lambda_2, dist.standard_normal_cdf(y1)) for y1 in y]
        index = index + 1

    ### scatter plot of the simulated defaults
    colors = ['blue', 'green', 'orange', 'red', 'yellow']

    mp = pu.PlotUtilities('Default Times with Correlations={0}'.format(rhos), 'x', 'y')

    mp.scatterPlot(tau_1, tau_2, rhos, colors)


def vasicek_large_portfolio_cdf(rho, p, x):

    nd = dist.NormalDistribution(0., 1.)
    y = (nd.inverse_cdf(x) * np.sqrt(1 - rho) - nd.inverse_cdf(p)) / np.sqrt(rho)
    return nd.cdf(y)


def plotVasicekDistribution(rhos, p, min_val, max_val, steps):

    x_label = 'x'
    y_label = 'CDF Value'
    chart_title = 'Vasicek Large Portfolio Distribution'
    step = (max_val - min_val) / steps
    x_ax = [min_val + step * k for k in range(steps)]
    y_ax = [[vasicek_large_portfolio_cdf(rho, p, x_val) for x_val in x_ax] for rho in rhos]

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multiPlot(x_ax, y_ax)


def conditional_default_prob(rho, p, z):
    nd = dist.NormalDistribution(0., 1.)
    y = (nd.inverse_cdf(p) - np.sqrt(rho) * z) / np.sqrt(1 - rho)
    return nd.cdf(y)

class functorConditionalLossDist():

    def __init__(self, rho, p, k, N, usePoissonApprox = False):

        self.p = p
        self.rho = rho
        self.k = k
        self.N = N
        self.usePoissonApprox = usePoissonApprox

    def pdf(self, z):

        p_z = conditional_default_prob(self.rho, self.p, z)
        if self.usePoissonApprox:
            return dist.poisson_pdf(self.N * p_z, self.k)
        else:
            return dist.binomial_pdf(p_z, self.k, self.N)

def portfolio_loss_histogram(rho, p, N, plotVsLHP = False):

    x_label = 'x'
    y_label = 'CDF Value'
    chart_title = 'Portfolio Loss Distribution'
    x_ax = [k for k in range(N+1)]
    pdf_v = [0.] * (N+1)

    for k in range(N+1):
        pdf_func = functorConditionalLossDist(rho, p, k, N, False)
        pdf_v[k] = gauss_hermite_integration_normalised(20, pdf_func.pdf)

    cdf_v = [sum([pdf_v[i] for i in range(k)]) for k in range(N+1)]

    tot_plot = 2 if plotVsLHP else 1

    plt.subplot(tot_plot, 1, 1)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(chart_title)

    plt.plot(x_ax, cdf_v, 'r*')

    if (plotVsLHP):
        plt.subplot(tot_plot, 1, 2)

        plt.xlabel('x')
        plt.ylabel('Large Portfolio Approximation')

        normed_ax = [float(x) / float(N) for x in x_ax]
        lp_val = [vasicek_large_portfolio_cdf(rho, p, x) for x in normed_ax]

        plt.plot(normed_ax, lp_val, 'b-')

    plt.show()
    plt.close()

def gauss_hermite_integration_normalised(deg, func):

    x, y = np.polynomial.hermite.hermgauss(deg)

    scaling_factor = 1./np.sqrt(np.pi)
    absc_factor = np.sqrt(2.)
    val = sum([func(absc_factor * x_i) * y_i for x_i, y_i in zip(x,y)]) * scaling_factor
    return val

def example_bivariate_option_price_mc(mean, variance, pd, pd_vol, strike, df):
    size = 15000
    standard_normal_sample = np.random.standard_normal(2 * size)
    ### we turn this into a random sample of dimension 2;

    vol = np.sqrt(variance)
    x = [mean * np.exp(vol * standard_normal_sample[k] - 0.5 * variance) for k in range(size)]
    y = [0. for k in range(size)]

    option_value = df * sum([max(x_0 - strike, 0) for x_0 in x]) / size
    print (option_value)
    min_rho = -0.99
    max_rho = 0.99
    step_size = 0.01
    rho_steps = int((max_rho - min_rho) / step_size)
    rhos = [min_rho + k * step_size for k in range(rho_steps)]
    option_values = [option_value for rho in rhos]

    mc_value = []
    default_threshold = dist.normal_CDF_inverse(pd) * pd_vol
    for rho in rhos:
        for k in range(size):
            z = pd_vol * (rho * standard_normal_sample[k] + np.sqrt(1 - rho * rho) * standard_normal_sample[k + size])
            y[k] = (0. if z <= default_threshold else 1.)
        mc_value.append(df * sum([y_0 * max(x_0 - strike, 0.) for y_0, x_0 in zip(y, x)]) / size)

    mp = pu.PlotUtilities('Risk-Adjusted Option Value As Function of Correlation', 'Correlation', 'Option Value')
    mp.multiPlot(rhos, [option_values, mc_value])


if __name__ == '__main__':


    size = 100

    rhos = [0.05, 0.1, 0.2, 0.5, 0.75]
    # -- portfolio loss distribution for finite case
    p = 0.05
    # portfolio_loss_histogram(rhos[2], p, 100, True)
    #
    # # -- portfolio loss distribution for LHP case
    # plotVasicekDistribution(rhos, p, 0., 0.25, 500)
    #
    # # -- demo of simple correlated defaults with different correlations
    # lambda_1 = 0.25
    # lambda_2 = 0.5
    # rhos = [0.95, 0.5, 0., -0.5, -0.95]
    # correlated_defaults_scatter(lambda_1, lambda_2, rhos, size)


    mean = 100
    variance = 0.2 * 0.2
    pd = 0.10
    pd_vol = 2.0
    strike = 105
    df = 1.0
    # example_bivariate_option_price_mc(mean, variance, pd, pd_vol, strike, df)
