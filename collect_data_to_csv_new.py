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

def read_info_from_hdf5(filename : str, required_input_info : list, required_output_info : list):

    try:
        info_file = h5py.File(filename)
    except FileNotFoundError:
        print("No such file or directory: {}".format(filename))
        return

    for key in info_file:
        if ("event" in key):
            # getting output data: first detector activation time
            # and multiplication of three activation probabilities

            min_activation_time = 999_999
            min_activation_key = ""
            required_detection_data = None

            if ("hits" in info_file[key].keys()):
                detectors_hits = info_file[key]["hits"]

                for hit in detectors_hits:
                    # get start of whole detection info
                    det_data = detectors_hits[hit]["data"][0]

                    # look for key with minimal activation time and memorize it
                    if (det_data[0] < min_activation_time):
                        min_activation_time = det_data[0]
                        min_activation_key = hit
            
                # adding data with full info
                required_detection_data = detectors_hits[min_activation_key]["data"][0]

            #print(required_detection_data)

            if (required_detection_data is not None):
                current_sample = [required_detection_data[0]]
                probs_data = required_detection_data[1:4]
                current_sample.append(probs_data[0] * probs_data[1] * probs_data[2])
                required_output_info.append(current_sample)
            else:
                continue # because theres no detectors activation

            # if there was output info we getting input data
            # it consist of particle energy, location and direction

            # assuming we have only one vertex and one particle
            event_vertex = info_file[key]['event_header']['vertices']['vertex0']

            # get location
            location_info = event_vertex["pos"][0:3]
            # get energy
            particle_energy = event_vertex["in_particles"]["particle0"][4]
            # get direction
            particle_direction = event_vertex["in_particles"]["particle0"][1:4]

            required_input_info.append([*location_info, *particle_direction, particle_energy])
    
    # some verification of supposed file structure...
    if (len(required_output_info) != len(required_input_info)):
        print("there a problem was appear: samples quantity is different, method worked not correctly")


def read_events_from_hdf5(filename : str, required_input_info : list, required_output_info : list):
    get_unified_prob = True

    try:
        info_file = h5py.File(filename)
    except FileNotFoundError:
        print("No such file or directory: {}".format(filename))
        return

    for key in info_file:
        if ("event" in key):
            
            current_hit_coords = None
            
            activation_time = None
            activation_prob = None

            particle_rel_coords = None
            particle_direction_vec = None
            particle_energy = None

            if ("hits" in info_file[key].keys()):
                on_hits_detectors = info_file[key]["hits"]
                
                for current_det_hits in on_hits_detectors:
                    current_det_data = on_hits_detectors[current_det_hits]["data"]
                    
                    for current_hit in current_det_data:
                        current_hit_coords = current_hit[5:8]
                        #print(current_hit_coords)

                        activation_time = current_hit[0]
                        activation_prob = current_hit[1:4]

                        output_sample = []
                        output_sample.append(activation_time)
                        if (get_unified_prob):
                            probs_multiplication = functools.reduce(lambda a, b: a*b, activation_prob)
                            output_sample.append(probs_multiplication)
                            #print(probs_multiplication)
                        else:
                            for prob in activation_prob:
                                output_sample.append(prob)
                        
                        event_vertex = info_file[key]['event_header']['vertices']['vertex0']
                        # get location
                        location_info = event_vertex["pos"][0:3]
                        # get energy
                        particle_energy = event_vertex["in_particles"]["particle0"][4]
                        # get direction
                        particle_direction = event_vertex["in_particles"]["particle0"][1:4]

                        # get relative location
                        rel_loc_info = location_info - current_hit_coords
                        
                        #print(location_info)
                        #print(current_hit_coords)
                        #print(location_info - current_hit_coords)

                        input_sample = [*rel_loc_info, *particle_direction, particle_energy]

                        #print(input_sample)
                        #print(output_sample)

                        required_input_info.append(input_sample)
                        required_output_info.append(output_sample)


def save_to_dataframe_csv(filename_1 : str, filename_2 : str, input_samples_list : list, output_samples_list : list):
    input_data_col_labels = ["rel_pos_x", "rel_pos_y", "rel_pos_z", 
                             "direct_x", "direct_y", "direct_z", 
                             "Energy"]

    output_data_col_labels = ["time_0", "activation_probability"]

    input_data_df = pd.DataFrame(input_samples_list, columns=input_data_col_labels)
    output_data_df = pd.DataFrame(output_samples_list, columns=output_data_col_labels)

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
        read_events_from_hdf5(current_relative_filename, required_input_info, required_output_info)

        time_stamp = int(time.time())
        default_input_data_csv_filename = default_out_folder + "/input-{}".format(time_stamp) + str(iteration_num) + ".csv"
        default_output_data_csv_filename = default_out_folder + "/output-{}".format(time_stamp) + str(iteration_num) + ".csv"
    
        save_to_dataframe_csv(default_input_data_csv_filename, default_output_data_csv_filename,
                              required_input_info, required_output_info)
