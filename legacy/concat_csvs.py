# small utility file
# very omnious concat method
# i think it should be made using pairs not whole collection

# TODO - perform reading csvs but without cc-*.csv files (which already concated)

import numpy as np
import pandas as pd
import time

def load_data_from_csv(filename : str):
    # loading and returning ndarray type
    data_frame = pd.read_csv(filename, index_col=0)
    return data_frame.to_numpy()


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


def save_to_dataframe_csv(filename_1 : str, filename_2 : str, input_samples_list : list, output_samples_list : list):
    input_data_col_labels = ["rel_pos_x", "rel_pos_y", "rel_pos_z", 
                             "direct_x", "direct_y", "direct_z", 
                             "Energy"]

    output_data_col_labels = ["time_0", "activation_probability"]

    input_data_df = pd.DataFrame(input_samples_list, columns=input_data_col_labels)
    output_data_df = pd.DataFrame(output_samples_list, columns=output_data_col_labels)

    input_data_df.to_csv(filename_1)
    output_data_df.to_csv(filename_2)


def clear_csvs(current_rel_folder_path = "./csv_data"):
    import os
    os.system("rm -f {}".format(current_rel_folder_path + "/input-*"))
    os.system("rm -f {}".format(current_rel_folder_path + "/output-*"))


def main():
    input_data = []
    output_data = []

    default_out_folder = "./csv_data"
    load_multiple_csvs(input_data, output_data, default_csv_data_folder=default_out_folder)

    # please, do not change filenames building logic
    # timestamps helps make (input-output) pair using idx after concat
    time_stamp = int(time.time())
    default_input_data_csv_filename = default_out_folder + "/cc_input-{}".format(time_stamp) + ".csv"
    default_output_data_csv_filename = default_out_folder + "/cc_output-{}".format(time_stamp) + ".csv"
    
    save_to_dataframe_csv(default_input_data_csv_filename, default_output_data_csv_filename,
                          input_data[0], output_data[0])

    clear_csvs()


if (__name__ == "__main__"):
    main()
