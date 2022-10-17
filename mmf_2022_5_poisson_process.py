################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################


import numpy as np

import core_math_utilities as dist
import plot_utilities as pu

def default_process_trajectories(intensity, sample_size):

   uni_sample = np.random.uniform(0., 1., sample_size)
   sampled_default_time = [dist.exponential_inverse_cdf(intensity, u) for u in uni_sample]
   max_time = 3. / intensity
   step_size = 0.01
   steps = int(max_time / step_size)
   x = [k * step_size for k in range(steps)]
   y = [[(0. if sdf > x_v else 1.) for x_v in x] for sdf in sampled_default_time]

   #######
   ### prepare and show plot
   ###
   mp = pu.PlotUtilities("Trajectories of Default Time Indicator With Intensity = {0}".format(intensity), 'Time', 'Default Indicator')
   mp.multiPlot(x, y)


def poisson_process(intensity, compensate):

   ## we are sampling the first sz jumps
   sz = 100
   uni_sample = np.random.uniform(0., 1., sz)

   #######
   ### transform the uniform sample to exponentials and subsequently into jumps
   #######
   sample = [dist.exponential_inverse_cdf(intensity,u) for u in uni_sample]
   jumps = [sum(sample[0:k]) for k in range(1,sz)]

   ####
   ## transform the jumps into trajectories of the counting process
   ####

   steps = 1000
   step_size = 0.05
   x = [k * step_size for k in range(steps)]

   y = [0] * steps

   for k in range(steps):
       for l in range(sz):
           if (jumps[l] > x[k]):
               y[k] = l
               break

   ### for future use - applying compensator of Poisson Process to turn into martingale....
   if compensate:
     y = [y_val - intensity * x_val for (y_val, x_val) in zip(y,x)]

   #######
   ### prepare and show plot
   ###
   mp = pu.PlotUtilities("Trajectory of Poisson Process with Intensity = {0}".format(intensity), 'Time', '# Of Jumps')
   mp.multiPlot(x, [y])

if __name__ == '__main__':

   intensity = 0.25
   compensate = False
   default_process_trajectories(intensity, 10)
   #poisson_process(intensity, compensate)