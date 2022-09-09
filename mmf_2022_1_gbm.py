################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################

import numpy as np
import plot_utilities as pu

###############
##
##  simple comparison of account value for different compounding frequencies
##
###############

def compounding_plot(rate, freq, min_val, max_val, steps):


   step = (max_val - min_val) / steps

   # values on the x axis
   x_ax = [min_val + step * k for k in range(steps)]

   ## container for y axis
   y_ax = [ [0.] * steps for f in freq]

   starting_value = 1.
   k = 0
   for f in freq:

       #######
       ### linear interpolation on a grid given by the compounding frequency
       #######
       __n = int((max_val - min_val) / f) + 1
       x_temp = [min_val + float(m) * f for m in range(__n)]
       y_temp = [starting_value] * __n
       for m in range(1, __n):
           y_temp[m] = y_temp[m-1] * (1. + rate * f)

       y_ax[k] = np.interp(x_ax, x_temp, y_temp)
       k = k + 1

   mp = pu.PlotUtilities('Compounding Account Value', 'x', 'Value')
   mp.multiPlot(x_ax, y_ax)

###############
##
##  adding a random shock to a fixed growth rate at any point
##
###############

def distorted_plot(rate, vols, min_val, max_val, steps):

   step = (max_val - min_val) / steps

   # values on the x axis
   x_ax = [min_val + step * k for k in range(steps)]

   ## container for y axis
   starting_value = 1.
   y_ax = [[starting_value for j in range(steps)] for vol in vols]

   k = 0

   for vol in vols:
        for m in range(1, steps):
            if (vol <= 0. ):
                random_shock = 0.
            else:
                random_shock = np.random.normal(0, vol * np.sqrt(step), 1)
            y_ax[k][m] = y_ax[k][m-1] * (1. + rate * step + random_shock)
        k = k + 1

   mp = pu.PlotUtilities('Compounding Account Value With Noise', 'x', 'Value')
   mp.multiPlot(x_ax, y_ax)

if __name__ == '__main__':


   rate = 0.1
   min_val = 0.
   max_val = 10.

   freq = [5, 2, 1, 0.1, 0.01, 0.001]
   steps = 100
   compounding_plot(rate, freq, min_val, max_val, steps)

   # vols = [0., 0.01, 0.05, 0.1, 1.0]
   vols = [0.1] * 10
   # vols[0] = 0.
   # steps = 1000
   # distorted_plot(rate, vols, min_val, max_val, steps)
