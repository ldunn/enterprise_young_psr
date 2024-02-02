# -*- coding: utf-8 -*-

from collections import OrderedDict

import numpy as np
from enterprise.signals import deterministic_signals, parameter, signal_base

# timing model delay


@signal_base.function
def tm_delay(residuals, t2pulsar, tmparams_orig, tmparams, phase=None, which='all', exclude=None, logs=None):
    """
    Compute difference in residuals due to perturbed timing model.

    :param residuals: original pulsar residuals from Pulsar object
    :param t2pulsar: libstempo pulsar object
    :param tmparams_orig: dictionary of TM parameter tuples, (val, err)
    :param tmparams: new timing model parameters, rescaled to be in sigmas
    :param which: option to have all or only named TM parameters varied

    :return: difference between new and old residuals in seconds
    """

    if which == 'all':
        keys = list(tmparams_orig.keys())
    else:
        keys = which

    if exclude is not None:
        for key in exclude:
            keys.remove(key)
    
    #print(which)
    # grab original timing model parameters and errors in dictionary
    orig_params = np.array([tmparams_orig[key] for key in keys])
    tmparams_copy = np.atleast_1d(tmparams).copy()

    # put varying parameters into dictionary
    tmparams_rescaled = np.atleast_1d(np.double(orig_params[:, 0] +
                                                tmparams_copy * orig_params[:, 1]))

    if logs is not None:
        for i in range(len(keys)):
            if any([x in keys[i] for x in logs]):
#                tmparams_rescaled[i] = np.double(10**(np.log10(orig_params[i, 0]) + (orig_params[i,1]-np.log10(orig_params[i,0]))*tmparams_copy[i]))
                tmparams_rescaled[i] = 10**(tmparams_copy[i]) - orig_params[i, 0]
                    
    tmparams_vary = OrderedDict(zip(keys, tmparams_rescaled))

    #print(tmparams_vary)
    # set to new values
    t2pulsar.vals(tmparams_vary)
    new_res = np.double(t2pulsar.residuals().copy())

    # remember to set values back to originals
    t2pulsar.vals(OrderedDict(zip(keys,
                                  np.atleast_1d(np.double(orig_params[:, 0])))))
    
    # Sort the residuals
    isort = np.argsort(t2pulsar.toas(), kind='mergesort')
    return residuals[isort] - new_res[isort] + (phase if phase is not None else 0)

# Model component building blocks #


def timing_block(tmparam_list=['RAJ', 'DECJ', 'F0', 'F1',
                               'PMRA', 'PMDEC', 'PX']):
    """
    Returns the timing model block of the model
    :param tmparam_list: a list of parameters to vary in the model
    """
    # default 5-sigma prior above and below the parfile mean
    tm_params = parameter.Uniform(-5.0, 5.0, size=len(tmparam_list))

    # timing model
    tm_func = tm_delay(tmparams=tm_params, which=tmparam_list)
    tm = deterministic_signals.Deterministic(tm_func, name='timing model')

    return tm
