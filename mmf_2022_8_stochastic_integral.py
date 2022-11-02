################
## Author: Thomas Balzer
## (c) 2022
## Material for MMF Stochastic Analysis - Fall 2022
################
import numpy as np
import plot_utilities as pu

def f_id(x):

    return x

def f_sq(x):

    return x * x

def f_exp(x):

    return np.exp(x)

def general_integration(integrand, integrator, upper_bound, scaling):

    n = upper_bound * scaling ### we will evaluate the integral t_0, \ldots, t_n
    dt = 1. / scaling
    x_values = [k * dt for k in range(0,n+1)]
    f_v = [integrand(x) for x in x_values]
    g_v = [integrator(x) for x in x_values]

    delta_g = [g2 - g1 for g1, g2 in zip(g_v[0:n], g_v[1:n+1])]
    integral = sum([ f*g for f, g in zip(f_v[0:n], delta_g)])

    return integral


def si_integrand_only(_steps, _paths, scaling):

    output_lhs_non_si = [0.] * (_paths)
    output_rhs_non_si = [0.] * (_paths)
    n = _steps * scaling
    delta_t = 1. / scaling
    delta_t_sq = 1. / np.sqrt(scaling)

    for m in range(_paths):

        ### create normal increments scaled to the right time step
        normal_sample = np.random.normal(0., 1., n)
        increments_bm = [s * delta_t_sq for s in normal_sample]

        ### create Brownian Motion paths
        bm_path = [0.] * (n+1)
        bm_path[1:n+1] = np.cumsum(increments_bm)
        output_lhs_non_si[m] = sum(bm_path[0:n]) * delta_t
        output_rhs_non_si[m] = sum(bm_path[1:n+1]) * delta_t

    num_bins = 50

    mp = pu.PlotUtilities('Stochastic Integral $\int_0^t B(u) du $ for 2 approximations', 'Outcome',
                          'Rel. Occurrence')

    colors = ['#0059ff', '#db46e2', '#ffc800', '#99e45e']

    mp.plotMultiHistogram([output_lhs_non_si, output_rhs_non_si], num_bins, colors)

def stochastic_integral_hist(_steps, _paths, scaling):

    output_lhs = [0.] * (_paths)
    output_rhs = [0.] * (_paths)
    n = _steps * scaling
    delta_t_sq = 1. / np.sqrt(scaling)

    for m in range(_paths):
        normal_sample = np.random.normal(0., 1., n)
        increments_bm = [s * delta_t_sq for s in normal_sample]

        ### create Brownian Motion paths
        bm_path = [0.] * (n+1)
        bm_path[1:n+1] = np.cumsum(increments_bm)

        output_lhs[m] = sum([f * g for f, g in zip(bm_path[0:n], increments_bm)])
        output_rhs[m] = sum([f * g for f, g in zip(bm_path[1:n+1], increments_bm)])

    num_bins = 50

    mp = pu.PlotUtilities('Stochastic Integral $\int_0^t B(u) dB(u) $ for 2 approximations', 'Outcome',
                          'Rel. Occurrence')

    colors = ['#0059ff', '#db46e2', '#ffc800', '#99e45e']

    mp.plotMultiHistogram([output_lhs, output_rhs], num_bins, colors)

if __name__ == '__main__':

    _paths = 5000
    _steps = 1
    scaling = 500
    # i = general_integration(f_exp, f_id, _steps, 100000)
    # print (i)
    # i = general_integration(f_exp, f_sq, _steps, 100000)
    # print (i)

    stochastic_integral_hist(_steps, _paths, scaling)
    si_integrand_only(_steps, _paths, scaling)
