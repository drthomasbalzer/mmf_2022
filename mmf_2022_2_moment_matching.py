################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################


import numpy as np
import matplotlib.pyplot as plt

import core_math_utilities as dist

###########
##
## Demo of Moment Matching
##
## - Sample a basket of a given size with random weights between (0,1)
## - Those basket constituents are assumed to follow a binomial distribution on $\{-1,+1\}$ with success probability $p$.
## - We then sample the outcome of the basket and compare it to a normal distribution with matching moments.
## - We can look at the distribution itself but also at tail probabilities
##
###########

def moment_matching(p, sz_basket):

   #####
   ## create a basket with random weights of size sz_basket
   #####

   lower_bound = 0.
   upper_bound = 1.

   weights = np.random.uniform(lower_bound, upper_bound, sz_basket)

   ## calculate mean and variance of the basket

   expectation = sum(weights) * dist.symmetric_binomial_expectation(p)
   variance = sum([w * w for w in weights]) * dist.symmetric_binomial_variance(p)

   simulation = 50000

   outcome = [0] * simulation

   for k in range(simulation):

       #######
       ### sample the basket constituents and determine the outcome for each trial
       #######
       uni_sample = np.random.uniform(lower_bound, upper_bound, sz_basket)
       sample = [dist.symmetric_binomial_inverse_cdf(p, u) for u in uni_sample]
       outcome[k] = sum([w * s for w,s in zip(weights, sample)])

   num_bins = 50

   plt.subplot(2,1,1)

   plt.title("Moment Matching of Binomial Basket of Size={0}".format(sz_basket))

   # the histogram of the data
   n, bins, _hist = plt.hist(outcome, num_bins, density=True, facecolor='blue', alpha=0.75)

   nd = dist.NormalDistribution(expectation, variance)

   mc_weight = 1./float(simulation)
   pdf_approx = [nd.pdf(b) for b in bins]
   call_option_approx = [nd.call_option(b) for b in bins]
   call_option_sampled = [ mc_weight * sum([max(oc - b, 0.) for oc in outcome]) for b in bins]

   plt.xlabel('Outcome')
   plt.ylabel('Rel. Occurrence')

   plt.plot(bins, pdf_approx, 'r*')

   plt.subplot(2,1,2)

   plt.xlabel('Call Option Strike')
   plt.ylabel('Call Option Price')

   plt.plot(bins, call_option_sampled, 'b-')
   plt.plot(bins, call_option_approx, 'r*')

   plt.show()

if __name__ == '__main__':

   moment_matching(0.75, 15)
