import h5py

f = h5py.File("./h5_coll/events1.h5")

for key in f:
    if ("event" in key and ("hits" in f[key].keys())):
        print("{} has hits.".format(key))
