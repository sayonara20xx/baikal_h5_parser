from math import sqrt
import numpy as np
import pandas as pd


def get_det_coords(detectors_coords_list : list):
    csv_data_default_folder = "../assist_data"
    csv_default_filename = "detector_coords.csv"

    full_relative_name = csv_data_default_folder + "/" + csv_default_filename
    csv_raw_data = pd.read_csv(full_relative_name, index_col=0)
    for row in csv_raw_data.iterrows():
        # `uid` is equal list idxs, dont need to add same info
        current_det = {"x": row[1][7], "y": row[1][8], "z": row[1][9]}
        detectors_coords_list.append(current_det)


def get_nearest_det(detectors_coords_list : list, hit_coords : dict):

    def get_distance(first_point : dict, second_point : dict):
        x1, y1, z1 = first_point.values()
        x2, y2, z2 = second_point.values()

        return float(sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2) + pow((z2 - z1), 2)))

    if ("x" in hit_coords and "y" in hit_coords and "z" in hit_coords and len(hit_coords.keys()) == 3):
        # decided to make naive search without external methods
        min_idx = -1
        min_value = 999_999_999
        for (idx, det) in zip(range(len(detectors_coords_list)), detectors_coords_list):
            curr_dist = get_distance(det, hit_coords)
            if (curr_dist < min_value):
                min_idx = idx
                min_value = curr_dist

        #print(min_idx)
        return min_idx

    else:
        print("Wrong hit coords input!\nOnly 3 followed keys needed: \"x\", \"y\", \"z\"")


def main():
    det_coords_list = []
    get_det_coords(det_coords_list)
    #print(det_coords_list)

    get_nearest_det(det_coords_list, {"x": 150, "y": 500, "z": 180})


if (__name__ == "__main__"):
    main()