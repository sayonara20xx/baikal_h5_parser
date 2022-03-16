# здесь надо будет написать код для анализа величин графически:
# гистограммы для всех значений вероятности и времени срабатывания, как начало
# потом их же, но после нормализации

import numpy as np


def normalize_data(data_samples_list : np.array):
    # у значений вероятности очень больной разброс значений даже после такой нормализации
    # возможно, нужен совсем другой способ
    data_array_collumns = data_samples_list.swapaxes(0,1)
    for col in data_array_collumns:
        t_median = np.median(col)
        t_max = np.max(col)
        t_min = np.min(col)
        for (i, elem) in zip(range(len(col)), col):
            # to [-1, 1]
            elem = (elem - t_median)/(t_max - t_min)
            col[i] = elem


def load_data_from_csv(filename : str):
    # loading and returning ndarray type
    import pandas as pd
    data_frame = pd.read_csv(filename, index_col=0)
    return data_frame.to_numpy()


def build_histogram(data : list, title : str = "histogram", n_bins : int = 10, param_range : tuple = (False, False)):
    import matplotlib.pyplot as plt

    # do not know how to fo same more reliable, sorry
    used_range = ()
    first, second = param_range
    if (first == second == False):
        used_range = (data.min(), data.max())
    else:
        used_range = param_range

    plt.hist(data, bins=n_bins, range=used_range)
    plt.title(title)
    plt.show()


def load_multiple_csvs(sample_data : list, default_csv_data_folder : str = "./csv_data"):
    '''
        returns nothing!\n
        fill input array with data
    '''
    import os
    csvs_filenames = os.listdir(default_csv_data_folder)
    
    numpy_arrays = []
    for csv_filename in csvs_filenames:
        relative_path = default_csv_data_folder + "/" + csv_filename
        if ("dataset" in csv_filename):
            current_array = load_data_from_csv(relative_path)
            numpy_arrays.append(current_array)

    sample_data.append(np.concatenate(numpy_arrays))


def main():
    sample_data = []
    load_multiple_csvs(sample_data, default_csv_data_folder="./csv_data")
    sample_data = sample_data[0]

    sample_data_swaped = sample_data.swapaxes(0, 1)
    # print(len(sample_data_swaped[0])) # ok

    data_cols_labels = ["target_det", "z", "rho", "theta", "phi"]
    for (col_label, i) in zip(data_cols_labels, range(len(data_cols_labels))):
        build_histogram(sample_data_swaped[i], n_bins=100, title=col_label)

    spec_data_cols_labels = {5 : "activation_time", 6: "probs_mult"}


if (__name__ == "__main__"):
    main()
