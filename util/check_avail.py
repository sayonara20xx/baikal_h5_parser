import h5py
import os

h5_rel_dir = "./h5_coll"
h5_files = os.listdir(h5_rel_dir)

for h5_file in h5_files:
	rel_h5_filename = h5_rel_dir + "/" + h5_file
	try:
		f = h5py.File(rel_h5_filename)
		print("file {} is OK".format(h5_file))
	except OSError:
		print("file {} is NOT OK".format(h5_file))
		os.system("rm {}".format(rel_h5_filename))
