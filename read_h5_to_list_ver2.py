'''
    primary file which processes h5 files after model
    working stage

    if new data is needed, add new columns in list,
    which `samples_list.append` refer

    do not forget also add according col label in .csv file
    ('save_to_dataframe_csv' method)
'''

from math import acos, sqrt
import h5py
import functools
import time
import pandas as pd
import numpy as np
from pandas._libs.tslibs import timestamps
import os

from detector_helper import get_det_coords
DETECTORS_COUNT = 864

def get_vec_module_2d(vec : list):
    return (sqrt(pow(vec[0], 2) + pow(vec[1], 2)))


def get_vec_module_3d(vec : list):
    return (sqrt(pow(vec[0], 2) + pow(vec[1], 2)) + pow(vec[2], 2))


def get_z(det_coords : list, cascade_coords : list):
    '''
        Z is just length of line between detector and
        cascade start position on Z axis
    '''
    return det_coords[2] - cascade_coords[2]


def get_rho(det_coords : list, cascade_coords : list):
    '''
        rho is distance between cascade start position
        and detector in XoY plane

        it's calculated using Pythagorean theorem and according coords difference
    '''
    return sqrt(pow(det_coords[0] - cascade_coords[0], 2) + pow(det_coords[1] - cascade_coords[1], 2))


def get_phi(det_coords : list, cascade_coords : list, cascade_vec : list):
    '''
        phi is angle between rho lane and cascade direction in XoY plane

        at first, i'm calculating vector with rho direction and normalize it
        by divide into it's module

        after, i'm calculating angle by formula with 'acos'

        rho (x1, y1)
        cascade (x2, y2)

                                x1 * x2 + y1 * y2
        alpha = acos _______________________________________
                      sqrt(x1^2 + y1^2) * sqrt(x2^2 + y2^2)
    '''
    two_dim_module = get_rho(det_coords, cascade_coords)
    rho_normalized_vector = [(det_coords[0] - cascade_coords[0]) / two_dim_module, 
                             (det_coords[1] - cascade_coords[1]) / two_dim_module]

    temp1 = rho_normalized_vector[0] * cascade_vec[0] + rho_normalized_vector[1] * cascade_vec[1]
    temp2 = get_vec_module_2d(rho_normalized_vector[:2]) * get_vec_module_2d(cascade_vec[:2])
    return acos(temp1 / temp2)


def get_theta(cascade_vec : list):
    '''
        theta is angle between Z and cascade direction
        z direction is (0, 0, 1)
        cascade (x1, y1, z1)
                                 z1
        alpha = acos __________________________
                      sqrt(x1^2 + y1^2 + z1^2)
    '''
    return acos(cascade_vec[2] / get_vec_module_3d(cascade_vec))


def read_hdf5(filename : str, samples_list : list):
    get_unified_prob = True
    probability_edge = 1*10e-20

    # determining if file exists
    try:
        info_file = h5py.File(filename)
    except FileNotFoundError:
        print("No such file or directory: {}".format(filename))
        return

    # prepare locations for detector determination method
    detectors_coords_list = []
    get_det_coords(detectors_coords_list)

    for key in info_file:
        if ("event" in key):
            
            # saving ids of activated dets
            mentioned_dets = []
            
            # at `event_vertex` key data about cascade angle, energy and start pos is located
            event_vertex = info_file[key]['event_header']['vertices']['vertex0']
            
            # get info from vertex
            location_info = event_vertex["pos"][0:3]
            particle_energy = event_vertex["out_particles"][0][4]
            particle_direction = list(event_vertex["out_particles"][0])[1:4]

            hit_coords = None

            if ("hits" in info_file[key].keys()):
                event_hits = info_file[key]["hits"]["data"]
                
                for current_hit in event_hits:
                    current_hit_info = list(current_hit)[0]
                    
                    on_hit_det_id = current_hit_info[0]
                    hit_time = current_hit_info[3]
                    hit_probs = current_hit_info[4:8]
                    hit_coords = current_hit_info[8:11]

                    if (on_hit_det_id not in mentioned_dets):
                        mentioned_dets.append(on_hit_det_id)

                    probs_multiplication = functools.reduce(lambda a, b: a*b, hit_probs)
                    if (probs_multiplication > probability_edge and probs_multiplication < 1):
                        phi = get_phi(hit_coords, location_info, particle_direction)
                        rho = get_rho(hit_coords, location_info)
                        theta = get_theta(particle_direction)
                        z = get_z(hit_coords, location_info)

                        samples_list.append([z, rho, theta, phi, probs_multiplication,
                                             on_hit_det_id])

            # loop works for each 'event' in h5 dataset
            for det_id in range(DETECTORS_COUNT):
                if (det_id not in mentioned_dets):
                    # writing same data for each unactivated det excludint act_time and probs_mult
                    det_center_coords = list(detectors_coords_list[det_id].values())

                    # these values don't requre hit info
                    phi = get_phi(hit_coords, location_info, particle_direction)
                    rho = get_rho(hit_coords, location_info)
                    theta = get_theta(particle_direction)
                    z = get_z(hit_coords, location_info)
                    
                    # append same way
                    samples_list.append([z, rho, theta, phi, None,
                                         det_id])

def debug_print(sample_list : list):
    for sample in sample_list:
        print(sample)


def save_to_dataframe_csv(filename : str, samples_list : list):
    data_cols_labels = ["z", "rho", "theta", 
                        "phi", "activation_time", "probs_mult"
                        "targer_det"]

    data_df = pd.DataFrame(samples_list, columns=data_cols_labels)
    data_df.to_csv(filename)


if (__name__ == "__main__"):
    default_data_folder_name = "./h5_coll"
    default_out_folder = "./csv_data"

    h5_filenames_list = os.listdir(default_data_folder_name)
    h5_filenames_list = list(filter(lambda x : (".h5" in x), h5_filenames_list))

    # im getting current iteration number to make unique file names for sure 
    for (iteration_num, current_filename) in zip(range(len(h5_filenames_list)), h5_filenames_list):
        sample_info = []

        current_relative_filename = default_data_folder_name + "/" + current_filename
        print("current h5 file relative name is {}".format(current_relative_filename))
        read_hdf5(current_relative_filename, sample_info)

        #debug_print(sample_info)
