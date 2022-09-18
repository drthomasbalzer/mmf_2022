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
##### Demo of Quantile Transformation
###########

## basic utility to create uniform sample

def getUniformSample(sz):

    lower_bound = 0.
    upper_bound = 1.

    sample = np.random.uniform(lower_bound, upper_bound, sz)

    return sample

## plot without transformation

def uniform_histogram(sz):

    uniform_sample = getUniformSample(sz)

    num_bins = 50
    hp = pu.PlotUtilities("Histogram of Uniform Sample of Size={0}".format(sz), 'Outcome', 'Rel. Occurrence')
    hp.plotHistogram(uniform_sample, num_bins)


#####
## Create distribution via Quantile Transform -- $B(1,p)$-distribution
#####

def binomial_histogram(p, sz):

    sample = [dist.binomial_inverse_cdf(p, u) for u in getUniformSample(sz)]
    num_bins = 100

    hp = pu.PlotUtilities("Histogram of Binomial Sample with Success Probability={0}".format(p), 'Outcome',
                          'Rel. Occurrence')
    hp.plotHistogram(sample, num_bins)


#####
## Create distribution via Quantile Transform -- $Exp(\lambda)$ distribution
#####

def exponential_histogram(_lambda, sz):

    sample = [dist.exponential_inverse_cdf(_lambda, u) for u in getUniformSample(sz)]
    num_bins = 50

    # the histogram of the data
    n, bins, _hist = plt.hist(sample, num_bins, density=True, facecolor='green', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Histogram of Exponential Sample with Parameter={0}".format(_lambda))

    y = [dist.exponential_pdf(_lambda, b) for b in bins]

    plt.plot(bins, y, 'r--')
    # # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()


#####
## Create distribution via Quantile Transform -- $B(1,p)$-distribution
#####

def normal_histogram(mu, var, sz):

    nd = dist.NormalDistribution(mu, var)
    #######
    ### transform the uniform sample
    #######
    sample = [nd.inverse_cdf(u) for u in getUniformSample(sz)]
    num_bins = 60

    hp = pu.PlotUtilities("Histogram of Normal Sample with Mean={0}, Variance={1}".format(mu, var), 'Outcome',
                          'Rel. Occurrence')
    hp.plotHistogram(sample, num_bins)


def lognormal_histogram(mu, var, sz):

    uniform_sample = getUniformSample(sz)

    nd = dist.NormalDistribution(0, var)
    #######
    ### transform the uniform sample
    #######
    ###
    strike = 70.
    sample = [''] * 2
    sample[0] = [mu * np.exp(nd.inverse_cdf(u) - 0.5 * var) for u in uniform_sample]
    sample[1] = [max(s - strike, 0.) for s in sample[0]]
    num_bins = 75

    hp = pu.PlotUtilities("Histogram of Lognormal Sample with Mean={0}, Variance={1}".format(mu, var), 'Outcome',
                          'Rel. Occurrence')
    hp.plotHistogram([sample[0]], num_bins)


def simulated_default_time(times, lambdas, sz):

    ### we need to build an interpolator of the integrated hazard rate
    y_values = [0.] * (len(times))
    for k in range(1, len(times)):
        y_values[k] = y_values[k-1] + (times[k] - times[k-1]) * lambdas[k]

    sampled_default_time = [np.interp(-np.log(1 - u), y_values, times) for u in getUniformSample(sz)]
    # sampled_default_time = [-np.log(1 - u) for u in getUniformSample(sz)]

    num_bins = 50

    plt.subplot(2, 1, 1)
    n, bins, _hist = plt.hist(sampled_default_time, num_bins, density=True, facecolor='green', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Simulated Default Time")

    y = [np.exp(-np.interp(b, times, y_values)) * np.interp(b, times, lambdas) for b in bins]
    plt.plot(bins, y, 'r*')
    # # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)

    plt.subplot(2, 1, 2)

    plt.xlabel('Time')
    plt.ylabel('Default Probability')

    pd = [1. - np.exp(-np.interp(b, times, y_values)) for b in bins]
    plt.plot(bins, pd, 'r-')

    plt.show()



if __name__ == '__main__':

    size = 50000

    times =  [0., 0.25, 1.0, 2.0, 4.0, 10., 100.]
    hr =  [0.01, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125]
    # hr =  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # hr =  [0.01, 0.025, 0.05, 0.075, 0.1, 0.125]
    #simulate_default_time(times, hr, size)

    calc_type = 0

    if (calc_type == 0):  ### uniform sample
        uniform_histogram(size)
    elif (calc_type == 1):  ### generate a binomial distribution
        p = 0.40
        binomial_histogram(p, size)
    elif (calc_type == 2):  ### generate an exponential distribution
        _lambda = 1.
        exponential_histogram(_lambda, size)
    elif (calc_type == 3):  ### generate a normal distribution
        mean = 30.
        variance = 0.2
        normal_histogram(mean, variance, size)
    elif (calc_type == 4):   ### generate a lognormal distribution
        mean = 100
        variance = 0.1
        lognormal_histogram(mean, variance, size)
    else:
        print ('Done')

