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
parser.add_argument('--par-dir', type=str, default="/fred/oz022/ldunn_glitch/UTMOST2D_analysis/enterprise/tn_pars")
parser.add_argument('--tim', type=str)
parser.add_argument('--idx', type=int)
parser.add_argument('--model', type=str, default="TNF2")
parser.add_argument('--fit-skypos', action='store_true')
parser.add_argument('--excise-tn-modes', action='store_true')
parser.add_argument('--out-dir', type=str, default="results_multinest/")
parser.add_argument('--equad-range', type=str)
parser.add_argument('--log10A-range', type=str)
parser.add_argument('--log-params', type=str)
parser.add_argument('--timing-param-range', type=float, default=5000)
parser.add_argument('--nlive', type=int, default=400)
parser.add_argument('--nmodes', type=int, default=30)

args = parser.parse_args()

DATA_DIR = "/fred/oz022/ldunn_glitch/UTMOST2D_analysis/enterprise"

if args.idx is not None:
    pars = sorted(glob.glob(f"{args.par_dir}/*.par"))
    idx = args.idx
    psr = pars[idx].split("/")[-1].split(".")[0]
    par = f"{args.par_dir}/{psr}.par"
    tim = f"{DATA_DIR}/pn_tims/{psr}.tim"
elif args.par is not None and args.tim is not None:
    par = args.par
    tim = args.tim
else:
    print("Must set either --idx or both of --par and --tim. Exiting.")
    exit(1)

psr = Pulsar(par, tim, drop_t2pulsar=False)
orig_resids = psr.t2pulsar.residuals()
psr.t2pulsar['RAJ'].fit = args.fit_skypos
psr.t2pulsar['DECJ'].fit = args.fit_skypos
#psr.t2pulsar['F2'].fit = False

#psr.t2pulsar.fit()

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

#log10_A_long = parameter.Uniform(-14,-5)("log10A_long")
#gamma_long = parameter.Uniform(0,10)("gamma_long")
##### Set up signals #####

selection = selections.Selection(selections.by_telescope)

# white noise
ef = white_signals.MeasurementNoise(efac=efac, log10_t2equad=log10_equad, selection=selection)

modes = np.array([n/tspan for n in range(1,args.nmodes)])
if args.excise_tn_modes:
    modes = np.delete(modes, np.where([x > 0.5/86400/365.25 and x < 2/86400/365.25 for x in modes]))
print(modes)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, modes=modes)

# red noise (powerlaw with 30 frequencies)
#modes, weights = model_utils.linBinning(1/tspan, 0, 1./10./tspan, 20, 10)
modes = np.array([n/2/tspan for n in range(1, 2*args.nmodes)])
if args.excise_tn_modes:
    modes = np.delete(modes, np.where([x > 0.5/86400/365.25 and x < 2/86400/365.25 for x in modes]))
#    weights = np.delete(weights, np.where([x > 0.5/86400/365.25 and x < 2/86400/365.25 for x in modes]))
print(modes)

#pl_long =  gp_priors.powerlaw_genmodes(log10_A=log10_A, gamma=gamma, wgts=weights)
#rn_long = gp_signals.FourierBasisGP(spectrum=pl_long, components=30, Tspan=tspan)
pl_long = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn_long = gp_signals.FourierBasisGP(spectrum=pl_long, modes=modes)

modes = np.array([2*n/tspan for n in range(1, args.nmodes//2)])
if args.excise_tn_modes:
    modes = np.delete(modes, np.where([x > 0.5/86400/365.25 and x < 2/86400/365.25 for x in modes]))
#    weights = np.delete(weights, np.where([x > 0.5/86400/365.25 and x < 2/86400/365.25 for x in modes]))
print(modes)

#pl_long =  gp_priors.powerlaw_genmodes(log10_A=log10_A, gamma=gamma, wgts=weights)
#rn_long = gp_signals.FourierBasisGP(spectrum=pl_long, components=30, Tspan=tspan)
pl_short = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn_short = gp_signals.FourierBasisGP(spectrum=pl_long, modes=modes)

log_modes = np.concatenate([np.logspace(np.log10(1/2/tspan), np.log10(1/tspan), 5), np.array([n/tspan for n in range(2,args.nmodes)])])
print(modes)
pl_log = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn_log = gp_signals.FourierBasisGP(spectrum=pl_log, modes=log_modes)

# timing model
#tm = gp_signals.TimingModel(use_svd=True)
#raj = parameter.Uniform(-10000, 10000)("RAJ")
#decj = parameter.Uniform(-10000, 10000)("DECJ")
#f0 = parameter.Uniform(-10000, 10000)("F0")
#f1 = parameter.Uniform(-10000, 10000)("F1")
#f2 = parameter.Uniform(-10000, 10000)("F2")

fit_params = [x for x in psr.tmparams_orig.keys()]
print(fit_params)

#log_params=["GLF0", "GLF0D", "GLTD"]
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
#timing_delay_f2_wf = timing.tm_delay(tmparams=f2_param, which=['F2'])
timing_delay_wf = timing.tm_delay(tmparams=timing_params, phase=phase_param, which=fit_params, logs=log_params)
timing_delay_nophase_wf = timing.tm_delay(tmparams=timing_params, phase=None, which=fit_params, logs=log_params)

timing = deterministic_signals.Deterministic(timing_delay_wf, name="timing_f2")
#timing_nof2 = deterministic_signals.Deterministic(timing_delay_nof2_wf, name="timing_nof2")

# full model is sum of components
#model_tnf2 = ef + tm + f2_signal + rn
#model_tn = ef + tm + rn

#model_f2 = ef + tm + f2_signal
#model_tnlongf2 = f2_signal + ef + tm + rn_long
#model_tnlong = ef + tm + rn_long
#model_nil = ef + tm

#model_tnlong = ef + timing_nof2 + rn_long
#model_tnlongf2 = ef + timing_nof2 + timing_f2 + rn_long
#model_tnf2 = ef + timing_nof2 + timing_f2 + rn
model_nil = ef + timing
model_tn = ef + timing + rn
model_tnlong = ef + timing + rn_long
model_tnshort = ef + timing + rn_short
model_tnlog = ef + timing + rn_log
#model_f2 = ef + timing_f2 + timing_nof2

if "TNLONG" in args.model:
    pta = signal_base.PTA([model_tnlong(psr)])
elif "TNLOG" in args.model:
    pta = signal_base.PTA([model_tnlog(psr)])
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
                    #log10base = np.log10(psr.t2pulsar[param_names[i]].val)
                    log10base = np.log10(psr.t2pulsar[param_names[i]].val)
                    params[i] = log10base + cube[i] * (-4 - log10base)
                elif "GLF0" in param_names[i]:
                    #log10base = np.log10(psr.t2pulsar[param_names[i]].val)
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

#sampler = ultranest.ReactiveNestedSampler(param_names, lnlike, prior_transform, log_dir=f"{args.out_dir}/{psrj}/{args.model}")
#sampler.stepsampler = ultranest.stepsampler.SliceSampler(
#        nsteps = 5*len(param_names),
#        adaptive_nsteps = 'move-distance',
#        generate_direction = ultranest.stepsampler.generate_mixture_random_direction)

#result = sampler.run(max_num_improvement_loops=0, min_num_live_points=args.nlive)
#sampler.print_results()
#sampler.plot()
#sampler.plot_trace()
