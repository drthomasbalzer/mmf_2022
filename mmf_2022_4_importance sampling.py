################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################



import numpy as np

import core_math_utilities as dist
import plot_utilities as pu


class OptionPayout():

    def __init__(self, strike):
        self.strike = strike

    def value(self, x):
        return 0.

class OptionPayoutCall(OptionPayout):

    def __init__(self, strike):
        self.strike = strike

    def value(self, x):
        return max(x - strike, 0.)


class OptionPayoutDigital(OptionPayout):

    def __init__(self, strike):
        self.strike = strike

    def value(self, x):
        if (x > strike):
            return 1.
        else:
            return 0.


def exponential_importance_sampling(_lambda, shift, strike, option_payout):

    ## we evaluate a payout of either $P(X > K)$ or $E(X-K)^+)$ for an exponential distribution with a given _lambda and a shifted model where
    ## the shift is set to bring the mean of the exponentially tilted distribution to the strike that is considered

    repeats = 500

    sample_post_is = [0.] * 2
    sample_post_is[0] = [0] * repeats  # this is be the sample WITHOUT I.S. applied
    sample_post_is[1] = [0] * repeats  # this is be the sample WITH I.S. applied

    for z in range(repeats):
        ## we are sampling sz times in each iteration

        sz = 5000

        lower_bound = 0.
        upper_bound = 1.

        ## we create a uniform sample first;

        uni_sample = np.random.uniform(lower_bound, upper_bound, sz)

        #######
        ### transform the uniform sample to exponentials with two different shifts
        #######

        sample_exp = [dist.exponential_inverse_cdf(_lambda, u) for u in uni_sample]
        sample_exp_shift = [dist.exponential_inverse_cdf(_lambda + shift, u) for u in uni_sample]

        ### evaluate the payout

        payout_non_is = 0
        payout_is = 0

        c = (_lambda + shift) / _lambda

        payout_non_is = sum([option_payout.value(s_e) for s_e in sample_exp]) / sz
        payout_is = sum([np.exp(shift * s_e_s) / c * option_payout.value(s_e_s) for s_e_s in sample_exp_shift]) / sz

        sample_post_is[0][z] = payout_non_is
        sample_post_is[1][z] = payout_is

    #######
    ### prepare and show plot
    ###
    num_bins = 50

    ### this is the exact result
    p = np.exp(-strike * _lambda)

    print (np.var(sample_post_is[0]))
    print (np.var(sample_post_is[1]))

    colors = ['green', 'blue']
    _title = "Option Payout with and w/o I.S."
    #    if (digitalPayout):
    #       _title =  "Digital Payout (Value = {0}) with and w/o I.S.".format(p)

    mp = pu.PlotUtilities(_title, 'Outcome', 'Rel. Occurrence')
    mp.plotMultiHistogram(sample_post_is, num_bins, colors)

def variance_normal_digital(strike):

    x_label = 'shift'
    y_label = 'Variance'
    chart_title = 'Sample Variance Under Exponential Tilting (Normal)'
    min_val = strike - 2.
    max_val = strike + 2.
    steps = 1000
    step = (max_val - min_val) / steps
    x_ax = [min_val + step * k for k in range(steps)]
    y_ax = [np.exp(x * x) * dist.standard_normal_cdf(-strike - x) for x in x_ax]

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multiPlot(x_ax, [y_ax])

def variance_exponential_digital(strike, intensity):

    _l = intensity
    x_label = 'shift'
    y_label = 'Variance'
    chart_title = 'Sample Variance Under Exponential Tilting (Exponential)'
    min_val = 0. # 1./strike * 0.2
    max_val = intensity - 0.001
    steps = 1000
    step = (max_val - min_val) / steps
    x_ax = [min_val + step * k for k in range(steps)]
    y_ax = [np.exp(-(_l + x) * strike) * _l * _l / (_l * _l - x * x) for x in x_ax]

    mp = pu.PlotUtilities(chart_title, x_label, y_label)
    mp.multiPlot(x_ax, [y_ax])

if __name__ == '__main__':
    intensity = 0.275
    strike = 15
    # shift = 1. / strike - intensity
    shift = 1. / strike - np.sqrt(1. / (strike * strike) + intensity * intensity)
    print (shift)
    p = OptionPayoutDigital(strike)
    # p = OptionPayoutCall(strike)
    #    digitalPayout = False
    exponential_importance_sampling(intensity, shift, strike, p)
    #variance_exponential_digital(strike, intensity)

    # variance_normal_digital(2.)
