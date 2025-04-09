import pickle as pkl

with open("data/spectra.pkl", "rb") as f:
    loaded_data = pkl.load(f)




print(loaded_data["13"])

print(len(loaded_data))