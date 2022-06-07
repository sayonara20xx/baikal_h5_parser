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

from detector_helper import get_det_coords, get_nearest_det


def fill_template(template : list):
    for i in range(876):
        template.append([-1, -1, -1, -1, 
                         -1, -1, -1, -1])


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

            # at `event_vertex` key data about cascade angle, energy and start pos is located
            event_vertex = info_file[key]['event_header']['vertices']['vertex0']
            # get info from vertex
            location_info = event_vertex["pos"][0:3]
            particle_energy = event_vertex["in_particles"]["particle0"][4]
            particle_direction = event_vertex["in_particles"]["particle0"][1:4]

            # we collecting data about not activated detectors also
            # using `hits` key we processing only activated
            # we need to save and store data about others
            # so on each event we create this list
            detectors_template = []
            fill_template(detectors_template)

            if ("hits" in info_file[key].keys()):
                on_hits_detectors = info_file[key]["hits"]
                
                for current_det_hits in on_hits_detectors:
                    current_det_data = on_hits_detectors[current_det_hits]["data"]
                    
                    for current_hit in current_det_data:
                        activation_prob = current_hit[1:5]
                        hit_coords = current_hit[5:8]
                        #print(activation_prob)

                        probs_multiplication = None
                        det_center_coords = None

                        if (get_unified_prob):
                            probs_multiplication = functools.reduce(lambda a, b: a*b, activation_prob)

                        if (probs_multiplication > probability_edge):
                            activation_time = current_hit[0]

                            # using assist method below to specify which det was activated
                            # unfortunately, no data about det is stored inside h5 files 
                            target_det = get_nearest_det(detectors_coords_list, {"x": hit_coords[0], "y": hit_coords[1], "z": hit_coords[2]})

                            # but we have data about detectors space location
                            # using hit and dets coords we able to specify det id we are looking for
                            det_center_coords = list(detectors_coords_list[target_det].values())
                            #print(probs_multiplication, target_det)
                    
                            # computating neccesary vars using for data analysis
                            phi = get_phi(det_center_coords, location_info, particle_direction)
                            rho = get_rho(det_center_coords, location_info)
                            theta = get_theta(particle_direction)
                            z = get_z(det_center_coords, location_info)

                            # verify detector determination
                            # some keys labeled det ID, but label is formed by unknown algorythm
                            #print(target_det, current_det_hits)

                            # appending dataset with new row
                            # adding new values also require adding new collumn label in saving method
                            samples_list.append([z, rho, theta, phi, activation_time, probs_multiplication,
                                                 target_det])


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
    # im getting current iteration number to make unique file names for sure 
    for (iteration_num, current_filename) in zip(range(len(h5_filenames_list)), h5_filenames_list):
        sample_info = []

        current_relative_filename = default_data_folder_name + "/" + current_filename
        print("current h5 file relative name is {}".format(current_relative_filename))
        read_hdf5(current_relative_filename, sample_info)

        #debug_print(sample_info)
