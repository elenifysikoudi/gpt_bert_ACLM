import h5py

dataf = open("../data/processed/segmented.txt", "r", encoding="utf-8")
data = []
for line in dataf:
    data.append(line.strip())


with h5py.File("../data/processed/all.hdf5", "w") as f:
    f.create_dataset("small10M", data=data)


