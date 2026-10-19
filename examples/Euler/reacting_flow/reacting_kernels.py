"""Five-species air thermodynamics and Park reaction rates used by this case."""

import numpy as np
from sympy import exp, log, tanh

RU = 8.314471468617452
MW = np.array([14.0067, 15.9994, 30.0061, 28.0134, 31.9988]) / 1000.0

# T1, T2, a1, a2, a3, b1, b2, b3 (NASA-9 coefficients).
THERMO = (
    (1000., 6000., [0.,0.,2.5,0.,0.,0.,0.], [8.876501380e4,-1.071231500e2,2.362188287,2.916720081e-4,-1.729515100e-7,4.012657880e-11,-2.677227571e-15], [5.475181050e8,-3.107574980e5,6.916782740e1,-6.847988130e-3,3.827572400e-7,-1.098367709e-11,1.277986024e-16], [5.610463780e4,4.193905036], [5.697351330e4,4.865231506], [2.550585618e6,-5.848769753e2]),
    (1000., 6000., [-7.953611300e3,1.607177787e2,1.966226438,1.013670310e-3,-1.110415423e-6,6.517507500e-10,-1.584779251e-13], [2.619020262e5,-7.298722030e2,3.317177270,-4.281334360e-4,1.036104594e-7,-9.438304330e-12,2.725038297e-16], [1.779004264e8,-1.082328257e5,2.810778365e1,-2.975232262e-3,1.854997534e-7,-5.796231540e-12,7.191720164e-17], [2.840362437e4,8.404241820], [3.392428060e4,-6.679585350e-1], [8.890942630e5,-2.181728151e2]),
    (1000., 6000., [-1.143916503e4,1.536467592e2,3.431468730,-2.668592368e-3,8.481399120e-6,-7.685111050e-9,2.386797655e-12], [2.239018716e5,-1.289651623e3,5.433936030,-3.656034900e-4,9.880966450e-8,-1.416076856e-11,9.380184620e-16], [-9.575303540e8,5.912434480e5,-1.384566826e2,1.694339403e-2,-1.007351096e-6,2.912584076e-11,-3.295109350e-16], [9.098214410e3,6.728725490], [1.750317656e4,-8.501669090], [-4.677501240e6,1.242081216e3]),
    (1000., 6000., [2.210371497e4,-3.818461820e2,6.082738360,-8.530914410e-3,1.384646189e-5,-9.625793620e-9,2.519705809e-12], [5.877124060e5,-2.239249073e3,6.066949220,-6.139685500e-4,1.491806679e-7,-1.923105485e-11,1.061954386e-15], [8.310139160e8,-6.420733540e5,2.020264635e2,-3.065092046e-2,2.486903333e-6,-9.705954110e-11,1.437538881e-15], [7.108460860e2,-1.076003744e1], [1.283210415e4,-1.586640027e1], [4.938707040e6,-1.672099740e3]),
    (1000., 6000., [-3.425563420e4,4.847000970e2,1.119010961,4.293889240e-3,-6.836300520e-7,-2.023372700e-9,1.039040018e-12], [-1.037939022e6,2.344830282e3,1.819732036,1.267847582e-3,-2.188067988e-7,2.053719572e-11,-8.193467050e-16], [4.975294300e8,-2.866106874e5,6.690352250e1,-6.169959020e-3,3.016396027e-7,-7.421416600e-12,7.278175770e-17], [-3.391454870e3,1.849699470e1], [-1.689010929e4,1.738716506e1], [2.293554027e6,-5.530621610e2]),
)

NU_F = np.array([[0,0,0,1,0],[0,0,0,0,1],[0,0,1,0,0],[0,1,0,1,0],[0,1,1,0,0]], dtype=int).T
NU_B = np.array([[2,0,0,0,0],[0,2,0,0,0],[1,1,0,0,0],[1,0,1,0,0],[1,0,0,0,1]], dtype=int).T
A_R = [3.0e16, 1.0e16, 5.0e9, 5.7e6, 8.4e6]
BETA_R = [-1.60, -1.50, 0.0, 0.42, 0.0]
THETA_R = [113200., 59360., 75500., 42938., 19400.]
ALPHA = np.array([[1.,1.,22.],[1.,1.,22.],[.2333,.20,22.],[.2333,.20,1.],[.2333,.20,1.]])


def _switches(temperature, t1, t2):
    low = 0.5 * tanh(-1.0e8 * (temperature - t1) / np.pi) + 0.5
    high = 0.5 * tanh(1.0e8 * (temperature - t2) / np.pi) + 0.5
    return low, 1.0 - low - high, high


def _enthalpy_polynomial(temperature, a, b):
    return (-a[0] / temperature**2 + a[1] * log(temperature) / temperature + a[2]
            + a[3] * temperature / 2 + a[4] * temperature**2 / 3
            + a[5] * temperature**3 / 4 + a[6] * temperature**4 / 5 + b[0] / temperature)


def _gibbs_polynomial(temperature, a, b):
    return (-a[0] / (2 * temperature**2) + a[1] * (log(temperature) + 1) / temperature
            + a[2] * (1 - log(temperature)) - a[3] * temperature / 2
            - a[4] * temperature**2 / 6 - a[5] * temperature**3 / 12
            - a[6] * temperature**4 / 20 + b[0] / temperature - b[1])


def enthalpy_over_rt(species, temperature):
    t1, t2, a1, a2, a3, b1, b2, b3 = species
    weights = _switches(temperature, t1, t2)
    return sum(weight * _enthalpy_polynomial(temperature, a, b)
               for weight, a, b in zip(weights, (a1, a2, a3), (b1, b2, b3)))


def gibbs_over_rt(species, temperature):
    t1, t2, a1, a2, a3, b1, b2, b3 = species
    weights = _switches(temperature, t1, t2)
    return sum(weight * _gibbs_polynomial(temperature, a, b)
               for weight, a, b in zip(weights, (a1, a2, a3), (b1, b2, b3)))


def pressure(temperature, densities):
    return temperature * sum(densities[i] / MW[i] for i in range(5)) * RU


def temperature_residual(temperature, molar_densities, internal_energy):
    value = -sum(molar_densities)
    value += sum(molar_densities[i] * enthalpy_over_rt(THERMO[i], temperature) for i in range(5))
    return value * temperature - internal_energy / RU


def production_rates(densities, temperature):
    molar = [densities[i] / MW[i] for i in range(5)]
    gibbs = [gibbs_over_rt(species, temperature) for species in THERMO]
    forward = [log(A_R[r]) + BETA_R[r] * log(temperature) - THETA_R[r] / temperature for r in range(5)]
    pressure_term = log(101325.0 / (RU * temperature))
    backward = [forward[r] + sum((NU_B[k,r] - NU_F[k,r]) * (gibbs[k] - pressure_term) for k in range(5)) for r in range(5)]
    third = [sum(ALPHA[k,r] * molar[k] for k in range(5)) if r < 3 else 1.0 for r in range(5)]
    reaction = []
    for r in range(5):
        cf = np.prod([molar[k] ** NU_F[k,r] for k in range(5)])
        cb = np.prod([molar[k] ** NU_B[k,r] for k in range(5)])
        reaction.append((exp(forward[r]) * cf - exp(backward[r]) * cb) * third[r])
    return np.array([MW[k] * sum((NU_B[k,r] - NU_F[k,r]) * reaction[r] for r in range(5)) for k in range(5)])
