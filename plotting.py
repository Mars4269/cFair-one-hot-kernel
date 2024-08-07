import matplotlib.pyplot as plt
import numpy as np

########## Results ##########

def plot_results(results, target, x_axis_description):

    correlations_none = [res[f'{target}_none'] for res in results]
    correlations_ones = [res[f'{target}_ones'] for res in results]
    correlations_random = [res[f'{target}_random'] for res in results]
    correlations_rf = [res[f'{target}_rf'] for res in results]
    cramers = [res[f'{target}_cramer'] for res in results]

    plt.plot(correlations_none, marker='o', linestyle='--', color='b', label='One Hot Kernel, a0 and b0 None')
    plt.plot(correlations_ones, marker='d', linestyle='--', color='r', label='One Hot Kernel, a0 and b0 ones')
    plt.plot(correlations_random, marker='d', linestyle='--', color='g', label='One Hot Kernel, a0 and b0 random')
    plt.plot(correlations_rf, marker='o', linestyle='--', color='y', label='One Hot Kernel, a0 and b0 rf')
    plt.plot(cramers, marker='o', linestyle='--', color='c', label='Cramer\'s V')
    
    plt.title(f'{target} plot')
    plt.xlabel(x_axis_description)
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_results_norm(results, target, x_axis_description):

    correlations_ones = [res[f'{target}_ones'] for res in results]
    correlations_ones_not_norm = [res[f'{target}_ones_not_norm'] for res in results]
    correlations_random = [res[f'{target}_random'] for res in results]
    correlations_random_not_norm = [res[f'{target}_random_not_norm'] for res in results]
    cramers = [res[f'{target}_cramer'] for res in results]

    plt.plot(correlations_ones, marker='o', linestyle='--', color='b', label='One Hot Kernel, a0 and b0 ones')
    plt.plot(correlations_ones_not_norm, marker='d', linestyle='--', color='r', label='One Hot Kernel, a0 and b0 ones not normalized')
    plt.plot(correlations_random, marker='d', linestyle='--', color='g', label='One Hot Kernel, a0 and b0 random')
    plt.plot(correlations_random_not_norm, marker='o', linestyle='--', color='y', label='One Hot Kernel, a0 and b0 random not normalized')
    plt.plot(cramers, marker='o', linestyle='--', color='c', label='Cramer\'s V')
    
    plt.title(f'{target} plot')
    plt.xlabel(x_axis_description)
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_results_delta(results, target, x_axis_description):

    correlations_none = [res[f'{target}_none'] for res in results]
    correlations_rf = [res[f'{target}_rf'] for res in results]
    cramers = [res[f'{target}_cramer'] for res in results]

    plt.plot(correlations_none, marker='o', linestyle='--', color='b', label='One Hot Kernel, a0 and b0 None with delta')
    plt.plot(correlations_rf, marker='o', linestyle='--', color='y', label='One Hot Kernel, a0 and b0 rf with delta')
    plt.plot(cramers, marker='o', linestyle='--', color='c', label='Cramer\'s V')
    
    plt.title(f'{target} plot')
    plt.xlabel(x_axis_description)
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)
    plt.show()    




def plot_results_general(results, target, x_axis_description):

    correlations_none = [res[f'{target}_none'] for res in results]
    correlations_rf = [res[f'{target}_rf'] for res in results]
    cramers = [res[f'{target}_cramer'] for res in results]

    plt.plot(correlations_none, marker='o', linestyle='--', color='b', label='One Hot Kernel, a0 and b0 None with delta')
    plt.plot(correlations_rf, marker='o', linestyle='--', color='y', label='One Hot Kernel, a0 and b0 rf without delta')
    plt.plot(cramers, marker='o', linestyle='--', color='c', label='Cramer\'s V')
        
    plt.title(f'{target} plot')
    plt.xlabel(x_axis_description)
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)
    plt.show()




def plot_results_hot_poly(results, target, x_axis_description):

    correlations_poly = [res[f'{target}_poly'] for res in results]
    correlations_rf = [res[f'{target}_rf'] for res in results]
    cramers = [res[f'{target}_cramer'] for res in results]

    plt.plot(correlations_poly, marker='o', linestyle='--', color='b', label='Polynomial Kernel, a0 and b0 None with delta')
    plt.plot(correlations_rf, marker='o', linestyle='--', color='y', label='One Hot Kernel, a0 and b0 rf without delta')
    plt.plot(cramers, marker='o', linestyle='--', color='c', label='Cramer\'s V')
    
    plt.title(f'{target} plot')
    plt.xlabel(x_axis_description)
    plt.ylabel(target)
    plt.legend()
    plt.grid(True)
    plt.show()




########## Coefficients ##########

def plot_coefficients(results, coeff, values, x_axis_description, absolute = True):
    
    ordered_val = np.unique(values)
    l = len(ordered_val)

    plt.figure(figsize=(20, 20))

    for i in range(l):

        c_none = [res[f'{coeff}_none'][i] for res in results]
        c_ones = [res[f'{coeff}_ones'][i] for res in results]       
        c_random = [res[f'{coeff}_random'][i] for res in results]
        c_rf = [res[f'{coeff}_rf'][i] for res in results]

        if absolute:
            c_none = list(map(abs, c_none))
            c_ones = list(map(abs, c_ones))
            c_random = list(map(abs, c_random))
            c_rf = list(map(abs, c_rf))

        plt.subplot(3, 3, i+1)
        
        plt.plot(c_none, marker='o', linestyle='--', color='b', label=f'{coeff} None')
        plt.plot(c_ones, marker='o', linestyle='--', color='r', label=f'{coeff} ones')
        plt.plot(c_random, marker='o', linestyle='--', color='g', label=f'{coeff} random')
        plt.plot(c_rf, marker='o', linestyle='--', color='y', label=f'{coeff} rf')

        plt.xlim(0, 30)
        plt.ylim(0, 1)

        plt.title(f'{coeff}{str(i+1)} - {ordered_val[i]}')
        plt.xlabel(x_axis_description)
        plt.ylabel(coeff)
        plt.legend()
        plt.grid(True)

    plt.show()




def plot_coefficients_norm(results, coeff, values, x_axis_description, absolute = True):
    
    ordered_val = np.unique(values)
    l = len(ordered_val)

    plt.figure(figsize=(20, 20))

    for i in range(l):

        c_ones = [res[f'{coeff}_ones'][i] for res in results]
        c_ones_not_norm = [res[f'{coeff}_ones_not_norm'][i] for res in results]       
        c_random = [res[f'{coeff}_random'][i] for res in results]
        c_random_not_norm = [res[f'{coeff}_random_not_norm'][i] for res in results]

        if absolute:
            c_ones = list(map(abs, c_ones))
            c_ones_not_norm = list(map(abs, c_ones_not_norm))
            c_random = list(map(abs, c_random))
            c_random_not_norm = list(map(abs, c_random_not_norm))

        plt.subplot(3, 3, i+1)
        
        plt.plot(c_ones, marker='o', linestyle='--', color='b', label=f'{coeff} ones')
        plt.plot(c_ones_not_norm, marker='o', linestyle='--', color='r', label=f'{coeff} ones not normalized')
        plt.plot(c_random, marker='o', linestyle='--', color='g', label=f'{coeff} random')
        plt.plot(c_random_not_norm, marker='o', linestyle='--', color='y', label=f'{coeff} random not normalized')

        plt.xlim(0, 30)
        plt.ylim(0, 1)

        plt.title(f'{coeff}{str(i+1)} - {ordered_val[i]}')
        plt.xlabel(x_axis_description)
        plt.ylabel(coeff)
        plt.legend()
        plt.grid(True)

    plt.show()



def plot_coefficients_delta(results, coeff, values, x_axis_description, absolute = True, zoom = None):
    
    ordered_val = np.unique(values)
    l = len(ordered_val)

    plt.figure(figsize=(20, 20))

    for i in range(l):

        c_none = [res[f'{coeff}_none'][i] for res in results]      
        c_rf = [res[f'{coeff}_rf'][i] for res in results]

        if absolute:
            c_none = list(map(abs, c_none))
            c_rf = list(map(abs, c_rf))

        plt.subplot(3, 3, i+1)
        
        plt.plot(c_none, marker='o', linestyle='--', color='b', label=f'{coeff} None with delta')
        plt.plot(c_rf, marker='d', linestyle='--', color='y', label=f'{coeff} rf with delta')

        plt.xlim(0, 30)
        if zoom != None:
            plt.ylim(0, zoom)
        else:
            plt.ylim(0, 1)

        plt.title(f'{coeff}{str(i+1)} - {ordered_val[i]}')
        plt.xlabel(x_axis_description)
        plt.ylabel(coeff)
        plt.legend()
        plt.grid(True)

    plt.show()




def plot_coefficients_general(results, coeff, values, x_axis_description, absolute = True, zoom = None):
    
    ordered_val = np.unique(values)
    l = len(ordered_val)

    plt.figure(figsize=(20, 20))

    for i in range(l):

        c_none = [res[f'{coeff}_none'][i] for res in results]      
        c_rf = [res[f'{coeff}_rf'][i] for res in results]

        if absolute:
            c_none = list(map(abs, c_none))
            c_rf = list(map(abs, c_rf))

        plt.subplot(3, 3, i+1)
        
        plt.plot(c_none, marker='o', linestyle='--', color='b', label=f'{coeff} None with delta')
        plt.plot(c_rf, marker='d', linestyle='--', color='y', label=f'{coeff} rf without delta')

        plt.xlim(0, 30)
        if zoom != None:
            plt.ylim(0, zoom)
        else:
            plt.ylim(0, 1)

        plt.title(f'{coeff}{str(i+1)} - {ordered_val[i]}')
        plt.xlabel(x_axis_description)
        plt.ylabel(coeff)
        plt.legend()
        plt.grid(True)

    plt.show()



def plot_coefficients_hot_poly(results, coeff, values, x_axis_description, absolute = True, zoom = None):
    
    ordered_val = np.unique(values)
    l = len(ordered_val)

    plt.figure(figsize=(20, 20))

    for i in range(l):

        c_poly = [res[f'{coeff}_poly'][i] for res in results]      
        c_rf = [res[f'{coeff}_rf'][i] for res in results]

        if absolute:
            c_poly = list(map(abs, c_poly))
            c_rf = list(map(abs, c_rf))

        plt.subplot(3, 3, i+1)
        
        plt.plot(c_poly, marker='o', linestyle='--', color='b', label=f'{coeff} polynomial, None with delta')
        plt.plot(c_rf, marker='d', linestyle='--', color='y', label=f'{coeff} one-hot, rf without delta')

        plt.xlim(0, 30)
        if zoom != None:
            plt.ylim(0, zoom)
        else:
            plt.ylim(0, 1)

        plt.title(f'{coeff}{str(i+1)} - {ordered_val[i]}')
        plt.xlabel(x_axis_description)
        plt.ylabel(coeff)
        plt.legend()
        plt.grid(True)

    plt.show()