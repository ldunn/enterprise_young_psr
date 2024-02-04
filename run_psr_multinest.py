from __future__ import division

import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.linalg as sl

import enterprise
from enterprise.pulsar import Pulsar,Tempo2Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from collections import OrderedDict
from enterprise.signals import gp_signals
from enterprise.signals import gp_priors
from enterprise.signals import deterministic_signals

import enterprise_extensions
from enterprise_extensions import hypermodel, model_utils
import timing

import corner

import pymultinest

import sys
import pathlib
import argparse

def parse_range(arg):
    return [float(arg.split(",")[0]), float(arg.split(",")[1])]

parser = argparse.ArgumentParser()

parser.add_argument('--par', type=str)
parser.add_argument('--tim', type=str)
parser.add_argument('--model', type=str, default="TNF2")
parser.add_argument('--fit-skypos', action='store_true')
parser.add_argument('--out-dir', type=str, default="results_multinest/")
parser.add_argument('--equad-range', type=str)
parser.add_argument('--log10A-range', type=str)
parser.add_argument('--log-params', type=str)
parser.add_argument('--timing-param-range', type=float, default=5000)
parser.add_argument('--nlive', type=int, default=400)
parser.add_argument('--nmodes', type=int, default=30)

args = parser.parse_args()

par = args.par
tim = args.tim

psr = Pulsar(par, tim, drop_t2pulsar=False)
orig_resids = psr.t2pulsar.residuals()
psr.t2pulsar['RAJ'].fit = args.fit_skypos
psr.t2pulsar['DECJ'].fit = args.fit_skypos

psr.t2pulsar['F2'].fit = "F2" in args.model
psr.t2pulsar['F2'].val = 0

rms_res = np.sqrt(np.mean(psr.t2pulsar.residuals()**2))*psr.t2pulsar['F0'].val
tspan = (psr.toas.max() - psr.toas.min())
psr.t2pulsar['F2'].err = 10*rms_res/(tspan)**3

psr.tmparams_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
for key in psr.tmparams_orig:
    psr.tmparams_orig[key] = (psr.t2pulsar[key].val,psr.t2pulsar[key].err)

for line in open(par, "r").readlines():
    line = line.strip().split()
    if line[0] == "PSRJ":
        psrj = line[1]

print(psr.tmparams_orig)

# Uniform prior on EFAC
efac_range = [0.1, 5.0]
efac = parameter.Uniform(efac_range[0], efac_range[1])

if args.equad_range is not None:
    equad_range = parse_range(args.equad_range)
else:
    equad_range = [-6, -2]
log10_equad = parameter.Uniform(equad_range[0], equad_range[1])

# red noise parameters 
# Uniform in log10 Amplitude and in spectral index
if args.log10A_range is not None:
    A_range = parse_range(args.log10A_range)
else:
    A_range = [-14, -7]
gamma_range = [2, 10]
log10_A = parameter.Uniform(A_range[0], A_range[1])
gamma = parameter.Uniform(gamma_range[0], gamma_range[1])

##### Set up signals #####

selection = selections.Selection(selections.by_telescope)

# white noise
ef = white_signals.MeasurementNoise(efac=efac, log10_t2equad=log10_equad, selection=selection)

# Red noise
modes = np.array([n/tspan for n in range(1,args.nmodes)])
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, modes=modes)

long_modes = np.array([n/2/tspan for n in range(1, 2*args.nmodes)])
pl_long = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn_long = gp_signals.FourierBasisGP(spectrum=pl_long, modes=long_modes)

short_modes = np.array([2*n/tspan for n in range(1, args.nmodes//2)])
pl_short = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn_short = gp_signals.FourierBasisGP(spectrum=pl_long, short_modes=modes)


fit_params = [x for x in psr.tmparams_orig.keys()]
print(fit_params)

if args.log_params is not None:
    log_params = args.log_params.split(",")
else:
    log_params = []

param_ranges_min = np.array([])
param_ranges_max = np.array([])
is_log = []
for param in fit_params:
    if any([x in param for x in log_params]):
        param_ranges_min = np.append(param_ranges_min, [np.log10(psr.t2pulsar[param].val)])
        param_ranges_max = np.append(param_ranges_max, [psr.t2pulsar[param].err])
        is_log.append(True)
    else:
        param_ranges_min = np.append(param_ranges_min, [-args.timing_param_range])
        param_ranges_max = np.append(param_ranges_max, [args.timing_param_range])
        is_log.append(False)

print(param_ranges_min)
print(param_ranges_max)

timing_params = parameter.Uniform(param_ranges_min, param_ranges_max, size=len(fit_params))
if "TNLONG" in args.model:
    phase_min = -25*np.max(np.abs(orig_resids))
else:
    phase_min = -25*np.max(np.abs(orig_resids))
    
phase_param = parameter.Uniform(phase_min, -phase_min)
timing_delay_wf = timing.tm_delay(tmparams=timing_params, phase=phase_param, which=fit_params, logs=log_params)

timing = deterministic_signals.Deterministic(timing_delay_wf, name="timing_f2")

model_nil = ef + timing
model_tn = ef + timing + rn
model_tnlong = ef + timing + rn_long
model_tnshort = ef + timing + rn_short
#model_f2 = ef + timing_f2 + timing_nof2

if "TNLONG" in args.model:
    pta = signal_base.PTA([model_tnlong(psr)])
elif "TNSHORT" in args.model:
    pta = signal_base.PTA([model_tnshort(psr)])
elif "TN" in args.model:
    pta = signal_base.PTA([model_tn(psr)])
else:
    pta = signal_base.PTA([model_nil(psr)])

#if args.model == "NIL":
#    pta = signal_base.PTA([model_nil(psr)])
#elif args.model == "TN":
#    pta = signal_base.PTA([model_tn(psr)])
#elif args.model == "F2":
#    pta = signal_base.PTA([model_f2(psr)])
#elif args.model == "TNF2":
#    pta = signal_base.PTA([model_tnf2(psr)])

num_params = len(pta.params)
print(num_params)
param_names = [p.name for p in pta.params[:-1]]
param_names += fit_params 

def prior_transform(cube):
    
    params = cube.copy()
    for i in range(len(param_names)):
        if "efac" in param_names[i]:
            params[i] = efac_range[0] + cube[i]*(efac_range[1] - efac_range[0])
        elif "equad" in param_names[i]:
            params[i] = equad_range[0] + cube[i]*(equad_range[1] - equad_range[0])
        elif "log10_A" in param_names[i]:
            params[i] = A_range[0] + cube[i]*(A_range[1] - A_range[0])
        elif "gamma" in param_names[i]:
            params[i] = gamma_range[0] + cube[i]*(gamma_range[1]-gamma_range[0])
        elif "phase" in param_names[i]:
            params[i] = phase_min + 2*(-phase_min)*cube[i]

        if param_names[i] in fit_params:
            if not any([log_param in param_names[i] for log_param in log_params]):
                params[i] = -args.timing_param_range + cube[i]*2*args.timing_param_range
            else:
                if "GLF0D" in param_names[i]:
                    log10base = np.log10(psr.t2pulsar[param_names[i]].val)
                    params[i] = log10base + cube[i] * (-4 - log10base)
                elif "GLF0" in param_names[i]:
                    log10base = np.log10(psr.t2pulsar[param_names[i]].val)
                    params[i] = log10base + cube[i] * (-4 - log10base)
                elif "GLTD" in param_names[i]:
                    log10base = np.log10(psr.t2pulsar[param_names[i]].val)
                    params[i] = log10base + cube[i]*(3 - log10base)
    return params

def lnlike(params):
    lnlike = pta.get_lnlikelihood(params)
    return lnlike

print(param_names)

out_dir = args.out_dir
pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

result = pymultinest.solve(LogLikelihood=lnlike, Prior=prior_transform, n_dims=len(param_names), outputfiles_basename=f"{out_dir}/", verbose=True, n_live_points=args.nlive, sampling_efficiency=0.3)

for name, col in zip(param_names, result['samples'].transpose()):
    print(f"{name}: {col.mean()} +- {col.std()}")


import json
with open(f"{out_dir}/params.json", 'w') as f:
    json.dump(param_names, f, indent=2)