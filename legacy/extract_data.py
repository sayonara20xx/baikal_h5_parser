# script will scan and read .h5 files from first folder (default name is "h5_coll")
# for each detected file input-{unix_time}.csv, output-{unix_time}.csv pair will created
# default out directory name is "cvs_data"
# info in csv's ready for import with `Pandas` lib, you can use `to_numpy` func

import h5py
import functools
import time
import pandas as pd
import numpy as np
from pandas._libs.tslibs import timestamps
import os

from detector_helper import get_det_coords, get_nearest_det


def read_hdf5(filename : str, input_samples_list : list, output_samples_list : list):
    get_unified_prob = True
    probability_edge = 1*10e-20

    try:
        info_file = h5py.File(filename)
    except FileNotFoundError:
        print("No such file or directory: {}".format(filename))
        return

    detectors_coords_list = []
    get_det_coords(detectors_coords_list)

    def fill_with_init_det_events(current_det_events_input : list, current_det_events_output : list, 
                                   location_info : list, particle_direction : list, particle_energy : float):
        det_count = 876
        for idx in range(det_count):
            current_row = [*location_info, *particle_direction, particle_energy]
            current_det_events_input.append(current_row)

            current_row = [-1, 0]
            current_det_events_output.append(current_row)

    for key in info_file:
        if ("event" in key):

            current_det_events_input = []
            current_det_events_output = []

            event_vertex = info_file[key]['event_header']['vertices']['vertex0']
            # get location
            location_info = event_vertex["pos"][0:3]
            # get energy
            particle_energy = event_vertex["in_particles"]["particle0"][4]
            # get direction
            particle_direction = event_vertex["in_particles"]["particle0"][1:4]            

            fill_with_init_det_events(current_det_events_input, current_det_events_output, location_info, particle_direction, particle_energy)

            if ("hits" in info_file[key].keys()):
                on_hits_detectors = info_file[key]["hits"]
                
                for current_det_hits in on_hits_detectors:
                    current_det_data = on_hits_detectors[current_det_hits]["data"]
                    
                    for current_hit in current_det_data:
                        activation_prob = current_hit[1:4]
                        probs_multiplication = -1.0

                        if (get_unified_prob):
                            probs_multiplication = functools.reduce(lambda a, b: a*b, activation_prob)
                        else:
                            print("Error! Not realised yet")
                            return

                        if (probs_multiplication > probability_edge):
                            activation_time = current_hit[0]
                            target_det = get_nearest_det(detectors_coords_list, {"x": location_info[0], "y": location_info[1], "z": location_info[2]})
                            current_det_events_output[target_det][0] = activation_time
                            current_det_events_output[target_det][1] = activation_prob
                            #print(probs_multiplication, target_det)

            #print(current_det_events_input, current_det_events_output)
            input_samples_list.append(current_det_events_input)
            output_samples_list.append(current_det_events_output)


def save_to_dataframe_csv(filename_1 : str, filename_2 : str, input_samples_list : list, output_samples_list : list):
    input_data_col_labels = ["rel_pos_x", "rel_pos_y", "rel_pos_z", 
                             "direct_x", "direct_y", "direct_z", 
                             "Energy"]

    output_data_col_labels = ["time_0", "activation_probability"]

    input_data_df = pd.DataFrame(input_samples_list, columns=input_data_col_labels)
    output_data_df = pd.DataFrame(output_samples_list, columns=output_data_col_labels)

    input_data_df.to_csv(filename_1)
    output_data_df.to_csv(filename_2)


def save_3D_to_dataframe_csv(filename_1 : str, filename_2 : str, input_samples_list : list, output_samples_list : list):
    #input_data_col_labels = ["rel_pos_x", "rel_pos_y", "rel_pos_z", 
    #                         "direct_x", "direct_y", "direct_z", 
    #                         "Energy"]

    #output_data_col_labels = ["time_0", "activation_probability"]

    input_data_df = pd.DataFrame(input_samples_list)
    output_data_df = pd.DataFrame(output_samples_list)

    input_data_df.to_csv(filename_1)
    output_data_df.to_csv(filename_2)


if (__name__ == "__main__"):
    default_data_folder_name = "./h5_coll"
    default_out_folder = "./csv_data"
    try:
        os.mkdir(default_data_folder_name)
    except FileExistsError:
        pass

    h5_filenames_list = os.listdir(default_data_folder_name)

    # im getting current iteration number to make unique file names for sure 
    for (iteration_num, current_filename) in zip(range(len(h5_filenames_list)), h5_filenames_list):
        required_input_info = []
        required_output_info = []

        current_relative_filename = default_data_folder_name + "/" + current_filename
        print("current h5 file relative name is {}".format(current_relative_filename))
        read_hdf5(current_relative_filename, required_input_info, required_output_info)

        # please, do not change filenames building logic
        # timestamps helps make (input-output) pair using idx after concat
        time_stamp = int(time.time())
        default_input_data_csv_filename = default_out_folder + "/input-{}".format(time_stamp) + str(iteration_num) + ".csv"
        default_output_data_csv_filename = default_out_folder + "/output-{}".format(time_stamp) + str(iteration_num) + ".csv"

        #print(required_input_info)
        save_3D_to_dataframe_csv(default_input_data_csv_filename, default_output_data_csv_filename,
                                 required_input_info, required_output_info)
