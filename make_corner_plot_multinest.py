#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.coordinates as coords
import astropy.units as u
import random
import glob
from tqdm.notebook import tqdm
import os
import libstempo
import libstempo.plot
import sys
from chainconsumer import ChainConsumer
import sys
from enterprise.pulsar import Pulsar, Tempo2Pulsar
from collections import OrderedDict

import enterprise_extensions
from enterprise_extensions import hypermodel, model_utils

import matplotlib
from matplotlib import rc

import json
import ultranest
import pandas as pd

import argparse

#matplotlib.rcParams["text.latex.preamble"] += r'\usepackage[dvips]{graphicx}\usepackage{amsmath}\usepackage{amssymb}'
plt.rcParams["figure.figsize"] = [8,6]
#rc('text', usetex=True)
plt.rcParams["font.size"] = 14.4
#sns.set(font_scale=1.2)
rc('font',**{'family':'serif'})
plt.rcParams['figure.facecolor'] = 'white'

parser = argparse.ArgumentParser()

parser.add_argument('--psrj', type=str)
parser.add_argument('--results-dir', type=str)
parser.add_argument('--in-par', type=str)
parser.add_argument('--in-tim', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--par-in-dir', type=str)
parser.add_argument('--par-out-dir', type=str)
parser.add_argument('--plot-dir', type=str)
parser.add_argument('--fit-skypos', action='store_true')
parser.add_argument('--output-tn-params', action='store_true')
parser.add_argument('--run', type=int)
parser.add_argument('--log-params', type=str)

args = parser.parse_args()

results_dir=args.results_dir
if args.model is None:
    model = results_dir.split("/")[-1]
else:
    model = args.model

burn_frac = 0.5

#log_params = ["GLF0", "GLF0D", "GLTD"]
if args.log_params is not None:
    log_params = args.log_params.split(",")
else:
    log_params = None

psrj = args.psrj

if args.in_par is None:
    par = f"{args.par_in_dir}/{psrj}.par"
else:
    par = args.in_par

if args.in_tim is None:
    tim = f"pn_tims/{psrj}.tim"
else:
    tim = args.in_tim

psr = Pulsar(par, tim, drop_t2pulsar=False)

psr.t2pulsar['RAJ'].fit = args.fit_skypos
psr.t2pulsar['DECJ'].fit = args.fit_skypos
#psr.t2pulsar['F2'].fit = False

#psr.t2pulsar.fit()

psr.t2pulsar['F2'].fit = "F2" in model
psr.t2pulsar['F2'].val = 0

rms_res = np.sqrt(np.mean(psr.t2pulsar.residuals()**2))*psr.t2pulsar['F0'].val
tspan = (psr.toas.max() - psr.toas.min())
psr.t2pulsar['F2'].err = 10*rms_res/(tspan)**3

psr_params = list(psr.t2pulsar.pars())
print(psr_params)
psr_params = OrderedDict.fromkeys(psr_params)
for key in psr_params:
    psr_params[key] = (psr.t2pulsar[key].val, psr.t2pulsar[key].err)


params_arr = np.array([psr_params[key] for key in psr_params.keys()])


sample_params = json.load(open(f"{results_dir}/params.json"))
samples = np.loadtxt(f"{results_dir}/post_equal_weights.dat")
samples = samples[:, :len(sample_params)]
print(sample_params)
print(psr_params.keys())
for i in range(len(sample_params)):
    param = sample_params[i]
    if param in psr_params.keys():
        if log_params is None or not any([log_param in param for log_param in log_params]):
            scaled_vals = [val*psr_params[param][1] for val in samples[:,i]]
        else:
#            scaled_vals = [10**(np.log10(psr_params[param][0]) + (psr_params[param][1] - np.log10(psr_params[param][0]))*val) for val in samples[param]]
            scaled_vals = [10**val - psr_params[param][0] for val in samples[:,i]]

        samples[:, i] = scaled_vals

samples = pd.DataFrame(samples, columns=sample_params)
c = ChainConsumer()
print(samples)
c.add_chain(samples)
c.plotter.plot()
plt.suptitle(f"{args.psrj} ({model})")
plt.savefig(f"{args.plot_dir}/{args.psrj}_{model}_corner.pdf")
