# здесь надо будет написать код для анализа величин графически:
# гистограммы для всех значений вероятности и времени срабатывания, как начало
# потом их же, но после нормализации


from functools import reduce
from typing import Counter
import numpy as np
from numpy.lib.index_tricks import s_


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


def load_multiple_csvs(input_data : list, output_data : list, default_csv_data_folder : str = "./csv_data"):
    '''
        returns nothing!\n
        fill two input arrays with data, packs it into list, use [0]
    '''
    import os
    csvs_filenames = os.listdir(default_csv_data_folder)
    
    input_numpy_arrays = []
    output_numpy_arrays = []
    for csv_filename in csvs_filenames:
        relative_path = default_csv_data_folder + "/" + csv_filename
        if ("input" in csv_filename):
            current_array = load_data_from_csv(relative_path)
            input_numpy_arrays.append(current_array)
        elif ("output" in csv_filename):
            current_array = load_data_from_csv(relative_path)
            output_numpy_arrays.append(current_array)

    input_data.append(np.concatenate(input_numpy_arrays))
    output_data.append(np.concatenate(output_numpy_arrays))


def main():
    input_data = []
    output_data = []

    load_multiple_csvs(input_data, output_data, default_csv_data_folder="./csv_data/det_direction_check")

    input_data_swapped = input_data[0].swapaxes(0,1)
    output_data_swapped = output_data[0].swapaxes(0,1)
    #print(len(input_data_swapped))

    build_histogram(input_data_swapped[0], n_bins=100, title="rel_pos_x")
    build_histogram(input_data_swapped[1], n_bins=100, title="rel_pos_y")
    build_histogram(input_data_swapped[2], n_bins=100, title="rel_pos_z")
    #build_histogram(input_data_swapped[3], n_bins=100, title="direct_x")
    #build_histogram(input_data_swapped[4], n_bins=100, title="direct_y")
    #build_histogram(input_data_swapped[5], n_bins=100, title="direct_z")
    #build_histogram(input_data_swapped[6], n_bins=100, title="energy")

    #build_histogram(output_data_swapped[0], n_bins=100, title="activation_time")
    build_histogram(output_data_swapped[1], n_bins=150, title="probability", param_range=(10e-6, 0.2))

    # очень кустарный метод расчёта степени логарифма числа
    # был бы здесь другой способ боже
    get_loga = output_data_swapped[1]
    logas = []
    for num in get_loga:
        string_num = "{:.50f}".format(float(num))
        cc = 0
        for c in string_num:
            if (c == '0' or c == '.'):
                cc += 1
            else:
                break
        
        #print(num)
        #print(cc - 1)
        logas.append(cc - 1)

    build_histogram(np.array(logas), n_bins=25, title="probability_exp",  param_range=(6, 0))

    #print(len(input_data_swapped[0]))


if (__name__ == "__main__"):
    main()
