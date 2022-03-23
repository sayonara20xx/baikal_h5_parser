# new concat method for 3dim data

# На будущее: вместо cvs надо было использовать pickle и DataFrame.to_pickle()

import os
import numpy as np
import pandas as pd
import time

from ast import literal_eval

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
    input_data_df = pd.DataFrame(input_samples_list)
    output_data_df = pd.DataFrame(output_samples_list)

    input_data_df.to_csv(filename_1)
    output_data_df.to_csv(filename_2)


def clear_csvs(current_rel_folder_path = "./csv_data"):
    import os
    os.system("rm -f {}".format(current_rel_folder_path + "/input-*"))
    os.system("rm -f {}".format(current_rel_folder_path + "/output-*"))


def main():
    input_data = []
    output_data = []

    default_out_folder = "./concated_data"

    try:
        os.mkdir(default_out_folder)
    except FileExistsError:
        pass

    load_multiple_csvs(input_data, output_data)

    print(np.array(input_data[0]).shape)
    #print(len(input_data))
    #print(len(input_data[0]))
    #print(len(input_data[0][1]))
    #result = input_data[0][1][0].strip('][').split(", ")
    #result = map(lambda x: float(x), result)
    #print(list(result))

    # Надеюсь я правильно помню, что ниже я пытался парсить 3-хмерные
    # данные из .csv, где списки по двум измерениям сохранялись в виде строки
    if (False):
        for sample in input_data[0][:1]:
            for current_detector_data in sample:
                # looks preety hard
                # just transforming list string representation into actual list of floats
                # i dont know the reason, but there are strings saved in .csv
                print(list(map(lambda x: float(x), current_detector_data.strip('][').split(", "))))

                # better use this
                print(literal_eval(current_detector_data))

        for sample in output_data[0][:100]:
            for current_detector_data in sample:
                if ('array' in current_detector_data):
                    temp_string = current_detector_data.strip('][)').split(", array([")
                    print([temp_string[0], *temp_string[1].split(", ")])
                else:
                    print(pd.eval(current_detector_data))


    #print(len(input_data[0][0]))
    time_stamp = int(time.time())
    default_input_data_csv_filename = default_out_folder + "/cc_input-{}".format(time_stamp) + ".csv"
    default_output_data_csv_filename = default_out_folder + "/cc_output-{}".format(time_stamp) + ".csv"
    
    save_to_dataframe_csv(default_input_data_csv_filename, default_output_data_csv_filename,
                          input_data[0], output_data[0])

    clear_csvs()


if (__name__ == "__main__"):
    main()