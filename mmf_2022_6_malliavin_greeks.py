################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu

import matplotlib.pyplot as plt

def plot_bachelier_option_price(start, vol):

    lower_bound = start - 10.
    upper_bound = start + 10.
    step = 0.01
    n_steps = int((upper_bound - lower_bound) / step)

    nd = dist.NormalDistribution(0., 1.)

    knock_out = 5

    x_ax = [lower_bound + k * step for k in range(n_steps)]
    y_ax = [ vol * nd.pdf( (max(x,knock_out) - start) / vol ) - (x - start) * nd.cdf( (start - max(x, knock_out)) / vol ) for x in x_ax]  # poisson distribution

    mp = pu.PlotUtilities("Bachelier Option Value as Function of Strike", "Option Strike", "Option Value")
    mp.multiPlot(x_ax, [y_ax], '-')


def malliavin_greeks(start, vol, strike, digitalPayout = False):

    ## we calculate the option value of a call option $(X-K)^+$ where the underlying is of the form $X = x_0 + sigma W$ with $W$ standard normal
    ## the aim is to calculate the sensitivities of the option price with respect to the x_0 both in bump and reval and with logarithmic Malliavin weights

    nd = dist.NormalDistribution(start, vol * vol)
    y = (start - strike) / vol
    if (digitalPayout):
        theo_option_price = nd.cdf(y)
        act_delta = dist.standard_normal_pdf(y) / vol
        act_gamma = - y * dist.standard_normal_pdf(y) / vol / vol
    else:
        theo_option_price = nd.call_option(strike)
        act_delta = dist.standard_normal_cdf(y)
        act_gamma = dist.standard_normal_pdf(y) / vol

    perturbation = 1.e-08

    # print (str("Theoretical Price: ") + str(theo_option_price))
    # print (str("Theoretical Delta: ") + str(act_delta))
    # print (str("Theoretical Gamma: ") + str(act_gamma))

    repeats = 500

    sample_delta = [0.] * 2
    sample_delta[0] = [0] * repeats  # this is the sample for the delta with B&R approach
    sample_delta[1] = [0] * repeats  # this is the sample for the delta with Malliavin logarithmic trick

    sample_gamma = [0.] * 2
    sample_gamma[0] = [0] * repeats # this is the sample for the gamma with B&R approach
    sample_gamma[1] = [0] * repeats # this is the sample for the gamma with Malliavin logarithmic

    sz = 5000
    total_sz = sz * repeats

    normal_sample = np.random.normal(0, 1, total_sz)

    for z in range(repeats):

        thisNormalSample = normal_sample[z * sz : (z+1) * sz]

        if (digitalPayout):
            option_value = sum([(0 if start + vol * ns < strike else 1.) for ns in thisNormalSample])

            malliavin_delta = sum([(0 if start + vol * ns < strike else 1.) * ns / vol for ns in thisNormalSample])
            malliavin_gamma = sum([(0 if start + vol * ns < strike else 1.) * (ns * ns - 1) / (vol * vol) for ns in thisNormalSample])

            option_value_pert = sum([(0 if start + perturbation + vol * ns < strike else 1.) for ns in thisNormalSample])
            option_value_pert_down = sum([(0 if start - perturbation + vol * ns < strike else 1.) for ns in thisNormalSample])
        else:
            option_value = sum([max(start + vol * ns - strike, 0.) for ns in thisNormalSample])

            malliavin_delta = sum([max(start + vol * ns - strike, 0.) * (ns) / (vol) for ns in thisNormalSample])
            malliavin_gamma = sum([max(start + vol * ns - strike, 0.) * (ns * ns - 1) / (vol * vol) for ns in thisNormalSample])

            option_value_pert = sum([max(start + perturbation + vol * ns - strike, 0.) for ns in thisNormalSample])
            option_value_pert_down = sum([max(start - perturbation + vol * ns - strike, 0.) for ns in thisNormalSample])


        sample_delta[0][z] = (option_value_pert - option_value) / sz / perturbation
        sample_delta[1][z] = malliavin_delta / sz
        sample_gamma[0][z] = (
                             option_value_pert - 2 * option_value + option_value_pert_down) / sz / perturbation / perturbation

        sample_gamma[1][z] = malliavin_gamma / sz

    #######
    ### prepare and show plot
    ###
    num_bins = 25

    print (np.var(sample_delta[0]))
    print (np.var(sample_delta[1]))


    plotGamma = False

    totalPlots = (2 if plotGamma else 1)

    ### Subplot for Delta Calculation
    plt.subplot(totalPlots, 1, 1)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("B + R vs Malliavin Delta (Value={0})".format(act_delta))

    n, bins, _hist = plt.hist(sample_delta[0], num_bins, density=True, facecolor='orange', alpha=0.5)
    n, bins, _hist = plt.hist(sample_delta[1], num_bins, density=True, facecolor='blue', alpha=0.75)

    ### Subplot for Gamma Calculation
    if (plotGamma):
        plt.subplot(totalPlots, 1, 2)

        plt.xlabel('Outcome')
        plt.ylabel('Rel. Occurrence')
        plt.title("B + R vs Malliavin Gamma (Value={0})".format(act_gamma))

        n, bins, _hist = plt.hist(sample_gamma[0], num_bins, density=True, facecolor='orange', alpha=0.55)
        n, bins, _hist = plt.hist(sample_gamma[1], num_bins, density=True, facecolor='blue', alpha=0.75)

    plt.show()


if __name__ == '__main__':
    start = 5.
    strike = 6.5
    vol = 1.0
    #plot_bachelier_option_price(start, vol)
    malliavin_greeks(start, vol, strike, True)

