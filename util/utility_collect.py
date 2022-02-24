# made this file only in validation purposes
# e.g. i need to check distribution of detector hits coords on OZ

import h5py
import functools
import time
import pandas as pd
import numpy as np
from pandas._libs.tslibs import timestamps
import os


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
                        rel_loc_info = current_hit_coords
                        
                        #print(location_info)
                        #print(current_hit_coords)
                        #print(location_info - current_hit_coords)

                        input_sample = [*rel_loc_info, *particle_direction, particle_energy]

                        #print(input_sample)
                        #print(output_sample)

                        required_input_info.append(input_sample)
                        required_output_info.append(output_sample)



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


if (__name__ == "__main__"):
    default_data_folder_name = "./h5_coll"
    default_out_folder = "./csv_data"
    current_filename = "auto_events_.h5"

    try:
        os.mkdir(default_data_folder_name)
    except FileExistsError:
        pass

    input_info = []
    output_info = []

    current_relative_filename = default_data_folder_name + "/" + current_filename
    print("current h5 file relative name is {}".format(current_relative_filename))
    read_events_from_hdf5(current_relative_filename, input_info, output_info)

    input_data_swapped = np.array(input_info).swapaxes(0,1)
    output_data_swapped = np.array(output_info).swapaxes(0,1)

    build_histogram(input_data_swapped[2], n_bins=100, title="hit_pos_z")