import numpy as np
import random as rn
import pandas as pd
import scipy.stats as ss
import time as t

from cfair.backends import NumpyBackend
from cfair.metrics.kernel.hgr import DoubleKernelHGR

backend = NumpyBackend()

warriors = ["Ultramarine", "Salamander", "White Scar", "Space Wolf", "Raven Guard", "Iron Hand", "Imperial Fist", "Blood Angel", "Dark Angel"]
animals = ["cat", "dog", "wolf", "horse"]

########## Cramer's V ##########

def cramers_v(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

########## General utils ##########

def one_hot_encode(x):
        unique_vals = np.unique(x)
        return np.array([[1 if val == xi else 0 for val in unique_vals] for xi in x]).transpose()

def sample(a_list):
    return rn.choice(a_list)

########## Kernels ##########

my_kernel_one_hot = DoubleKernelHGR(
    backend=backend,          
    kernel_a=one_hot_encode, 
    kernel_b=one_hot_encode, 
)

my_kernel_one_hot_with_delta = DoubleKernelHGR(
    backend=backend,          
    kernel_a=one_hot_encode, 
    kernel_b=one_hot_encode, 
    delta_independent=0.2
)

polynomial_kernel = DoubleKernelHGR(
    backend=backend,          
    kernel_a=9, 
    kernel_b=4, 
    delta_independent=0.2
)

########## Experiments ##########

def compute_corr_and_exec_time(feat1, feat2, kernel = "cramer", init = None, norm = True):

    if kernel == "cramer":
        start_time = t.time()
        confusion_matrix = pd.crosstab(feat1, feat2)
        result = cramers_v(confusion_matrix.values)
        end_time = t.time()
        exec_time = end_time - start_time

    else:
        l1, l2 = len(np.unique(feat1)), len(np.unique(feat2))

        a0, b0 = None, None # Default case is None

        if init == "ones":
            a0 = np.ones(l1)
            b0 = np.ones(l2)
            if norm:
                a0 = a0/l1
                b0 = b0/l2

        elif init == "random":
            a0=np.random.rand(l1)
            b0=np.random.rand(l2)
            if norm:
                a0 = a0/a0.sum()
                b0 = b0/b0.sum()

        elif init == "rel_fr":
            a0 = np.array([ feat1.count(f1_value)/len(feat1) for f1_value in np.unique(feat1)])
            b0 = np.array([ feat2.count(f2_value)/len(feat2) for f2_value in np.unique(feat2)])

        start_time = t.time()
        result = kernel._result(feat1, feat2, kernel_a=True, kernel_b=True, a0=a0, b0=b0)
        end_time = t.time()
            
        exec_time = end_time - start_time

    return result, exec_time




def experiment(dataset_creation_function, kernel):
    results = []

    for i in range(30):

        res_i = {}

        war_i, ani_i = dataset_creation_function()

        result_i_none, exec_time_none = compute_corr_and_exec_time(war_i, ani_i, kernel)

        result_i_ones, exec_time_ones = compute_corr_and_exec_time(war_i, ani_i, kernel, "ones")

        result_i_random, exec_time_random = compute_corr_and_exec_time(war_i, ani_i, kernel, "random")

        result_i_rf, exec_time_rf = compute_corr_and_exec_time(war_i, ani_i, kernel, "rel_fr")

        cramer, exec_time_cramer = compute_corr_and_exec_time(war_i, ani_i)

        res_i['correlation_none'] = result_i_none.value
        res_i['alpha_none'] = result_i_none.alpha
        res_i['beta_none'] = result_i_none.beta
        res_i['time_none'] = exec_time_none

        res_i['correlation_ones'] = result_i_ones.value
        res_i['alpha_ones'] = result_i_ones.alpha
        res_i['beta_ones'] = result_i_ones.beta
        res_i['time_ones'] = exec_time_ones

        res_i['correlation_random'] = result_i_random.value
        res_i['alpha_random'] = result_i_random.alpha
        res_i['beta_random'] = result_i_random.beta
        res_i['time_random'] = exec_time_random

        res_i['correlation_rf'] = result_i_rf.value
        res_i['alpha_rf'] = result_i_rf.alpha
        res_i['beta_rf'] = result_i_rf.beta
        res_i['time_rf'] = exec_time_rf

        res_i['correlation_cramer'] = cramer
        res_i['time_cramer'] = exec_time_cramer

        results.append(res_i)

    return results




def experiment_norm(dataset_creation_function, kernel):
    results = []

    for i in range(30):

        res_i = {}

        war_i, ani_i = dataset_creation_function()

        result_i_ones, exec_time_ones = compute_corr_and_exec_time(war_i, ani_i, kernel, "ones")
        result_i_ones_not_norm, exec_time_ones_not_norm = compute_corr_and_exec_time(war_i, ani_i, kernel, "ones", False)

        result_i_random, exec_time_random = compute_corr_and_exec_time(war_i, ani_i, kernel, "random")
        result_i_random_not_norm, exec_time_random_not_norm = compute_corr_and_exec_time(war_i, ani_i, kernel, "random", False)

        cramer, exec_time_cramer = compute_corr_and_exec_time(war_i, ani_i)

        res_i['correlation_ones'] = result_i_ones.value
        res_i['alpha_ones'] = result_i_ones.alpha
        res_i['beta_ones'] = result_i_ones.beta
        res_i['time_ones'] = exec_time_ones

        res_i['correlation_ones_not_norm'] = result_i_ones_not_norm.value
        res_i['alpha_ones_not_norm'] = result_i_ones_not_norm.alpha
        res_i['beta_ones_not_norm'] = result_i_ones_not_norm.beta
        res_i['time_ones_not_norm'] = exec_time_ones_not_norm

        res_i['correlation_random'] = result_i_random.value
        res_i['alpha_random'] = result_i_random.alpha
        res_i['beta_random'] = result_i_random.beta
        res_i['time_random'] = exec_time_random

        res_i['correlation_random_not_norm'] = result_i_random_not_norm.value
        res_i['alpha_random_not_norm'] = result_i_random_not_norm.alpha
        res_i['beta_random_not_norm'] = result_i_random_not_norm.beta
        res_i['time_random_not_norm'] = exec_time_random_not_norm

        res_i['correlation_cramer'] = cramer
        res_i['time_cramer'] = exec_time_cramer

        results.append(res_i)

    return results




def experiment_delta(dataset_creation_function, kernel):
    results = []

    for i in range(30):

        res_i = {}

        war_i, ani_i = dataset_creation_function()

        result_i_none, exec_time_none = compute_corr_and_exec_time(war_i, ani_i, kernel)

        result_i_rf, exec_time_rf = compute_corr_and_exec_time(war_i, ani_i, kernel, "rel_fr")

        cramer, exec_time_cramer = compute_corr_and_exec_time(war_i, ani_i)

        res_i['correlation_none'] = result_i_none.value
        res_i['alpha_none'] = result_i_none.alpha
        res_i['beta_none'] = result_i_none.beta
        res_i['time_none'] = exec_time_none

        res_i['correlation_rf'] = result_i_rf.value
        res_i['alpha_rf'] = result_i_rf.alpha
        res_i['beta_rf'] = result_i_rf.beta
        res_i['time_rf'] = exec_time_rf

        res_i['correlation_cramer'] = cramer
        res_i['time_cramer'] = exec_time_cramer

        results.append(res_i)

    return results




def experiment_general(dataset_creation_function, kernel1, kernel2):
    results = []

    for i in range(30):

        res_i = {}

        war_i, ani_i = dataset_creation_function(i)

        result_i_none, exec_time_none = compute_corr_and_exec_time(war_i, ani_i, kernel1)

        result_i_rf, exec_time_rf = compute_corr_and_exec_time(war_i, ani_i, kernel2, "rel_fr")

        cramer, exec_time_cramer = compute_corr_and_exec_time(war_i, ani_i)

        res_i['correlation_none'] = result_i_none.value
        res_i['alpha_none'] = result_i_none.alpha
        res_i['beta_none'] = result_i_none.beta
        res_i['time_none'] = exec_time_none

        res_i['correlation_rf'] = result_i_rf.value
        res_i['alpha_rf'] = result_i_rf.alpha
        res_i['beta_rf'] = result_i_rf.beta
        res_i['time_rf'] = exec_time_rf

        res_i['correlation_cramer'] = cramer
        res_i['time_cramer'] = exec_time_cramer

        results.append(res_i)

    return results




def experiment_hot_poly(dataset_creation_function, kernel1, kernel2):
    results = []

    for i in range(30):

        res_i = {}

        war_i, ani_i = dataset_creation_function()

        result_i_poly, exec_time_poly = compute_corr_and_exec_time(war_i, ani_i, kernel1)

        result_i_rf, exec_time_rf = compute_corr_and_exec_time(war_i, ani_i, kernel2, "rel_fr")

        cramer, exec_time_cramer = compute_corr_and_exec_time(war_i, ani_i)

        res_i['correlation_poly'] = result_i_poly.value
        res_i['alpha_poly'] = result_i_poly.alpha
        res_i['beta_poly'] = result_i_poly.beta
        res_i['time_poly'] = exec_time_poly

        res_i['correlation_rf'] = result_i_rf.value
        res_i['alpha_rf'] = result_i_rf.alpha
        res_i['beta_rf'] = result_i_rf.beta
        res_i['time_rf'] = exec_time_rf

        res_i['correlation_cramer'] = cramer
        res_i['time_cramer'] = exec_time_cramer

        results.append(res_i)

    return results