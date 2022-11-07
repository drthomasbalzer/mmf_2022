################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################


import numpy as np

import plot_utilities as pu

########
## -- Generic Base Class of C^{1,2} functions with derivatives where Ito's formula applies
########

class FunctionC12():

   def __init__(self, funct):
     self.funct = funct

   def value(self, _t, _x):
       return self.funct.value(_x)

   def f_t(self, _t, _x):
       return 0.

   def f_x(self, _t, _x):
       return self.funct.f_x(_x)

   def f_xx(self, _t, _x):
       return self.funct.f_xx(_x)

class Function1D():

   def value(self, _x):
       return 0.

   def f_x(self, _x):
       return 0.

   def f_xx(self, _x):
       return 0.

########
## -- F(t,x) = x^{\alpha}
########

class genericMonomial(Function1D):

    def __init__(self, _alpha):
        self.alpha = _alpha

    def value(self, _x):
        return np.power(_x, self.alpha)

    def f_x(self, _x):
        return self.alpha * np.power(_x, self.alpha - 1)

    def f_xx(self, _x):
        return self.alpha * (self.alpha - 1) * np.power(_x, self.alpha - 2)

########
## -- F(t,x) = \exp(x)
########

class exponentialFunction(Function1D):

   def value(self, _x):
       return np.exp(_x)

   def f_x(self, _x):
       return self.value(_x)

   def f_xx(self, _x):
       return self.value(_x)

########
## -- F(t,x) = \ln(x)
########

class logarithmicFunction(Function1D):

   def value(self, _x):
       return np.log(_x)

   def f_t(self, _x):
       return 0.

   def f_x(self, _x):
       return 1. / _x

   def f_xx(self, _x):
       return - 1. / (_x * _x)


class itoProcess:

   def __init__(self, a, b, x):

       ## standard Ito representation of a process
       ## X(t) = X(0) + \int_0^t a(u, X(u)) du + \int_0^t b(u, X(u)) dB(u)

       self.drift = a
       self.diffusion = b
       self.initial_value = x

   def a(self, _t, _x):
       return 0.

   def b(self, _t, _x):
       return 0.

class itoProcessExp(itoProcess):

   def a(self, _t, _x):
       return self.drift * _x

   def b(self, _t, _x):
       return self.diffusion * _x

class itoProcessStandard(itoProcess):

   def a(self, _t, _x):
       return self.drift

   def b(self, _t, _x):
       return self.diffusion


def plot_ito_process(_maxTime, _timestep, _number_paths, itoPr, funct):

   ## normals for a single paths
   funcWrapper = FunctionC12(funct)
   size = int(_maxTime / _timestep)

   ## total random numbers needed
   total_sz = size * _number_paths

   sample = np.random.normal(0, np.sqrt(_timestep), total_sz)

   paths = [0.] * (_number_paths)

   x = [_timestep * k for k in range(size + 1)]\

   ####
   ## plot the trajectory of the Ito process
   ####
   i = 0
   for k in range(_number_paths):
       drivingProcess = itoPr.initial_value
       v = funcWrapper.value(0, itoPr.initial_value)
       path = [v] * (size + 1)
       for j in range(1, size + 1):
           ########
           ## the paths will be constructed through application of Ito's formula
           ########

           _x = drivingProcess
           _u = x[j-1]
           _du = _timestep
           _a = itoPr.a(_u, _x)
           _b = itoPr.b(_u, _x)

           ####
           ## -- path of actual F(t, X(t))
           ## F(t + dt, X(t + dt)) = F(t, X(t)) + F_t(t,X(t)) a(t,x) dt
           ##      + F_x(t, X(t)) a(t,x) dt + \frac{1}{2} F_xx(t,X(t)) b^2(t,x) dt
           ##      + F_x(t, X(t)) b(t,x) dB(t)
           ####

           path[j] = ( path[j - 1] + funcWrapper.f_t(_u, _x ) * _du
                       + _a * funcWrapper.f_x(_u, _x) * _du
                       + 0.5 * _b * _b * funcWrapper.f_xx(_u, _x) * _du
                       + _b * funcWrapper.f_x(_u, _x) * sample[i] )

           ####
           ## -- underlying ito process
           ## X(t + dt) = X(t) + a(t,x) dt + b(t,x) dB(t)
           ####

           drivingProcess = drivingProcess + _a * _du + _b * sample[i]

           ### increment counter for samples

           i = i + 1

       paths[k] = path


   #######
   ### prepare and show plot
   ###

   mp = pu.PlotUtilities('Paths of Ito Process $F(t,X(t))$', 'Time', 'Random Walk Value')
   mp.multiPlot(x, paths)


if __name__ == '__main__':

   max_time = 5
   timestep = 0.001
   paths = 12

   f_id = genericMonomial(1.)  ### generic form of identity function
   f_exp = exponentialFunction()
   f_sq = genericMonomial(2.)  ### generic form of square function
   f_ln = logarithmicFunction()
   f_gm = genericMonomial(.1)

   vol = 0.2

   drift_0 = 0.05
   iPS = itoProcessStandard(drift_0, vol, 0.)

   drift_1 = 0.0
   iPE = itoProcessExp(drift_1, vol, 1.)

   plot_ito_process(max_time, timestep, paths, iPE, f_sq)