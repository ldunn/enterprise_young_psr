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
parser.add_argument('--in-par', type=str)
parser.add_argument('--in-tim', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--results-dir', type=str)
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
psr_params = OrderedDict.fromkeys(psr_params)
for key in psr_params:
    psr_params[key] = (psr.t2pulsar[key].val, psr.t2pulsar[key].err)


params_arr = np.array([psr_params[key] for key in psr_params.keys()])

sample_params = json.load(open(f"{results_dir}/params.json"))
samples = np.loadtxt(f"{results_dir}/phys_live.points")
print(sample_params)
print(psr_params.keys())
for i in range(len(sample_params)):
    param = sample_params[i]
    if param in psr_params.keys():
        mean = np.mean(samples[:,i])
        err = np.std(samples[:,i])
        if log_params is None or not any([log_param in param for log_param in log_params]):
            scaled_val = psr_params[param][0] + np.mean([val*psr_params[param][1] for val in samples[:,i]])
        else:
#            log10_val = np.log10(psr_params[param][0]) + (psr_params[param][1] - np.log10(psr_params[param][0]))*np.mean(samples[param])
            print(param)
            print(samples[:,i])
            scaled_val = np.mean(10**samples[:,i]) - psr_params[param][0]


        scaled_err = err*psr_params[param][1]
        psr.t2pulsar[param].val = scaled_val
        psr.t2pulsar[param].err = scaled_err

psr.t2pulsar.savepar(f"{args.par_out_dir}/{psrj}_{model}.par")

with open(f"{args.par_out_dir}/{psrj}_{model}.par", "a") as f:
    print(f"RAJ {psr.t2pulsar['RAJ'].val*24/2/np.pi}", file=f)
    print(f"DECJ {psr.t2pulsar['DECJ'].val*360/2/np.pi}", file=f)

    if args.output_tn_params:
        for i in range(len(sample_params)):
            param = sample_params[i]
            if "mo_efac" in param:
                mean = np.mean(samples[:,i])
                print(f"TNEF -B mo {mean}", file=f)
            if "mons_efac" in param:
                mean = np.mean(samples[:,i])
                print(f"TNEF -B mons {mean}", file=f)
            if "mo_log10_t2equad" in param:
                mean = np.mean(samples[:,i])
                print(f"TNEQ -B mo {mean}", file=f)
            if "mons_log10_t2equad" in param:
                mean = np.mean(samples[:,i])
                print(f"TNEQ -B mons {mean}", file=f)
            if "log10_A" in param:
                mean = np.mean(samples[:,i])
                print(f"TNRedAmp {mean}", file=f)
            if "gamma" in param:
                mean = np.mean(samples[:,i])
                print(f"TNRedGam {mean}", file=f)
                if "TNLONG" in model:
                    print(f"TNRedC 60", file=f)
                elif "TNSHORT" in model:
                    print(f"TNRedC 15", file=f)
                elif "TN" in model:
                    print(f"TNRedC 30", file=f)
                if "TNLONG" in model:
                    print(f"TNRedFLow {np.log10(0.5)}", file=f)
                if "TNSHORT" in model:
                    print(f"TNRedFLow {np.log10(2)}", file=f)

plt.errorbar(psr.toas/86400, psr.t2pulsar.residuals(), yerr=1e-6*psr.t2pulsar.toaerrs, fmt='.', alpha=0.2)
plt.suptitle(f"{psrj} ({model})")
if args.plot_dir is not None:
    plt.savefig(f"{args.plot_dir}/{psrj}_{model}.pdf")
    plt.clf()
