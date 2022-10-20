################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################



import numpy as np

import plot_utilities as pu

class BMFunctor():

    def __init__(self, vol):
        self.vol = vol

    def nextSample(self, prevVal, sample):
        return 0.

    def initialValue(self):
        return 0.

    def type(self):
        return 'None'


class BrownianMotion(BMFunctor):

    def nextSample(self, prevVal, sample):
        return prevVal + sample * self.vol

    def initialValue(self):
        return 0.

    def type(self):
        return 'Brownian Motion'

class GeometricBrownianMotion(BMFunctor):

    def nextSample(self, prevVal, sample):
        return prevVal * np.exp(self.vol * sample - 0.5 * self.vol * self.vol)

    def initialValue(self):
        return 1.

    def type(self):
        return 'Geometric Brownian Motion'

def geometric_brownian_motion(_time, _timestep, _number_paths, bM):

    size = int(_time / _timestep)
    total_sz = size * _number_paths

    sample = np.random.normal(0, 1, total_sz)

    paths = [0.] * _number_paths
    max_paths = [0.] * _number_paths

    # set up x-axis
    x = [_timestep * k for k in range(size + 1)]

    ####
    ## plot the trajectory of the process
    ####
    i = 0
    for k in range(_number_paths):
        path = [bM.initialValue()] * (size + 1)
        # max_path = [bM.initialValue()] * (size + 1)
        for j in range(size + 1):
            if (j == 0):
                continue ## nothing
            else:
                path[j] = bM.nextSample(path[j-1], sample[i])
                # max_path[j] = max(max_path[j - 1], path[j])
                i = i + 1

        paths[k] = path
        # max_paths[k] = max_path

    max_paths = [[max(path[0:j]) for j in range(1,len(path)+1)] for path in paths]
    # max_paths = [max(path[0:j]) for j in range(1,len(path)+1) for path in paths]
    # print (max_paths)
    # max_paths = [ [ max([path[j] for j in range(n)]) for n in range(len(path))] for path in paths]

    mp = pu.PlotUtilities(r'Paths of ' + str(bM.type()), 'Time', 'Random Walk Value')

    plot_max = True
    if (plot_max):
        plot_all_max = False
        if (plot_all_max):
            mp.multiPlot(x, max_paths)
        else: # only the first path and its running maximum
            thesePaths = [0.] * 2
            thesePaths[0] = paths[0]
            thesePaths[1] = max_paths[0]
            mp.multiPlot(x, thesePaths)
    else:
        mp.multiPlot(x, paths)


if __name__ == '__main__':


    paths = [[0.01, 0.02, 0.01, 0.04]]
    # # print (path[0:0])
    #
    # n = 4
    # # max_paths = [path[0:j] for j in range(1,n+1)]
    # #
    # # for m in max_paths:
    # #     print (max(m))
    # #
    # # print (max_paths)
    # max_paths = [[max(path[0:j]) for j in range(1,len(path)+1)] for path in paths]
    # print (max_paths)
    #


    time = 5
    timestep = 0.001
    paths = 1
    volatility = 0.2
    bM = BrownianMotion(volatility * np.sqrt(timestep))
#    bM = GeometricBrownianMotion(volatility * np.sqrt(timestep))
    geometric_brownian_motion(time, timestep, paths, bM)


