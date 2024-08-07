import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_hist(dataset, feature, relative_feature = None, relative_value = None):

    title = f'{feature} distribution'

    if relative_feature!=None:
        
        title = title + f' relative to {relative_feature} equal to {relative_value}'
        dataset_rel = dataset[ dataset[relative_feature] == relative_value ]

        freq_total = dataset[feature].value_counts(normalize=True).reset_index()
        freq_total.columns = [feature, 'relative_freq']
        freq_total['dataset'] = 'Full dataset'

        freq_relative = dataset_rel[feature].value_counts(normalize=True).reset_index()
        freq_relative.columns = [feature, 'relative_freq']
        freq_relative['dataset'] = f'Relative to {relative_feature} = {relative_value}'

        # Combine the frequencies
        combined_freq = pd.concat([freq_total, freq_relative])

        # Plot using seaborn
        plt.figure(figsize=(20,10))
        sns.barplot(data=combined_freq, x=feature, y='relative_freq', hue='dataset')
        plt.title(f'Relative Frequencies of {feature}')
        plt.ylabel('Relative Frequency')
        plt.xlabel(feature)
        plt.show()
        
    else:
        plt.figure(figsize=(20,10))

        sns.countplot(data=dataset, x=feature, order=np.unique(dataset[feature]), stat="proportion")
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.title(title)

        plt.show()



def compute_correlation(dataset, feat1, feat2, kernel):
    
    sorted_feat1 = np.unique(dataset[feat1])
    sorted_feat2 = np.unique(dataset[feat2])


    feat1_rf_a0 = np.array([ len(dataset[ dataset[feat1] == value ])/len(dataset) for value in sorted_feat1])
    feat2_rf_b0 = np.array([ len(dataset[ dataset[feat2] == value ])/len(dataset) for value in sorted_feat2])
   
    return kernel._result(dataset[feat1], dataset[feat2], kernel_a=True, kernel_b=True, a0=feat1_rf_a0, b0=feat2_rf_b0)




def my_sort(couple):
    return abs(couple[1]) 




def print_coefficients(values, coefficients):

    max_len = max(len(value) for value in values)

    to_print = []
    
    for value, coefficient in zip(values, coefficients):
        to_print.append((value, abs(coefficient)))

    to_print.sort(key=my_sort, reverse=True)

    for couple in to_print:
        print("{:<{width}} : {}".format(couple[0], couple[1], width=max_len))


    

def print_coefficients_alphabetical(values, coefficients):

    max_len = max(len(value) for value in values)

    for value, coefficient in zip(values, coefficients):
        print("{:<{width}} : {}".format(value, coefficient, width=max_len))


    
