# resolves issues with very big probs in dataset!

import os
import numpy as np
import pandas as pd
import time


def load_data_from_csv(filename : str):
    # loading and returning ndarray type
    data_frame = pd.read_csv(filename, index_col=0)
    return data_frame.to_numpy()


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


def save_to_dataframe_csv(sample_data : list, default_out_data_folder : str = ""):
    data_cols_labels = ["target_det", "z", "rho", "theta", 
                        "phi", "activation_time", "probs_mult"]

    if (not bool(default_out_data_folder)):
        print("No output concated data folder specified! using \"./concat_data/dataset-000.csv\"")
        default_out_data_folder = "./concat_data/dataset-000.csv"

    data_df = pd.DataFrame(sample_data, columns=data_cols_labels)
    data_df.to_csv(default_out_data_folder)


def filter_csv_data(sample_data : list):
    for (i, sample) in zip(range(len(sample_data)), sample_data):
        if (sample[6] >= 1.0):
            sample_data.pop(i)
            i -= 1


def main():
    default_data_folder = "./csv_data"
    csv_filenames_list = os.listdir(default_data_folder)

    for filename in csv_filenames_list:
        if ("aaa" in filename):
            current_file_fullname = default_data_folder + "/" + filename
            data_samples = load_data_from_csv(current_file_fullname)
            filter_csv_data(data_samples)
            save_to_dataframe_csv(data_samples, current_file_fullname)


if (__name__ == "__main__"):
    main()