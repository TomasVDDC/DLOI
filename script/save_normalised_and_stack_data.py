import numpy as np
import pathlib

# Path to the data
path_to_data = "Data/Train + Test/"
path_to_save_normalised_data = "Data/normalised/"
path_to_save_min_max_of_labels = "Data/normalised/min_max_labels/"   # Save the min and max values of the labels so that we can denormalize them later


pathlib.Path(path_to_data).mkdir(parents=True, exist_ok=True) 
pathlib.Path(path_to_save_normalised_data).mkdir(parents=True, exist_ok=True)
pathlib.Path(path_to_save_min_max_of_labels).mkdir(parents=True, exist_ok=True)


#load the data
X_BC_train = np.load(path_to_data + "X_BC_train.npy", allow_pickle=True, mmap_mode="r")
X_M_train = np.load(path_to_data + "X_M_train.npy", allow_pickle=True, mmap_mode="r")
X_BC_test = np.load(path_to_data + "X_BC_test.npy", allow_pickle=True, mmap_mode="r")
X_M_test = np.load(path_to_data + "X_M_test.npy", allow_pickle=True, mmap_mode="r")
X_BC_val = np.load(path_to_data + "X_BC_val.npy", allow_pickle=True, mmap_mode="r")
X_M_val = np.load(path_to_data + "X_M_val.npy", allow_pickle=True, mmap_mode="r")

labels_train = np.load(path_to_data + "labels_train.npy", allow_pickle=True, mmap_mode="r")
labels_test = np.load(path_to_data + "labels_test.npy", allow_pickle=True, mmap_mode="r")
labels_val = np.load(path_to_data + "labels_val.npy", allow_pickle=True, mmap_mode="r")


# Only keep first 500 samples for faster training
num_samples = 500
X_BC_train = X_BC_train[:num_samples]
X_M_train = X_M_train[:num_samples]
X_BC_test = X_BC_test[:num_samples]
X_M_test = X_M_test[:num_samples]
X_BC_val = X_BC_val[:num_samples]
X_M_val = X_M_val[:num_samples]


print(f"- Normalizing and saving the first {num_samples} samples of the data")


# Normalize the data
X_BC_train_normalised = X_BC_train / 255.0
X_M_train_normalised = X_M_train / 255.0
X_BC_test_normalised = X_BC_test / 255.0
X_M_test_normalised = X_M_test / 255.0
X_BC_val_normalised = X_BC_val / 255.0
X_M_val_normalised = X_M_val / 255.0


# Stack the data because model architecture requires it
X_train_normalised = np.stack((X_BC_train_normalised, X_M_train_normalised), axis=1)
X_test_normalised = np.stack((X_BC_test_normalised, X_M_test_normalised), axis=1)
X_val_normalised = np.stack((X_BC_val_normalised, X_M_val_normalised), axis=1)

# Normalize the labels
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data,min_val,max_val

# Apply normalization and save the min/max values
labels_train_normalised, min_labels_train, max_labels_train = min_max_normalize(labels_train)
labels_test_normalised, min_labels_test, max_labels_test = min_max_normalize(labels_test)
labels_val_normalised,  min_labels_val, max_labels_val = min_max_normalize(labels_val)

print("- Data normalized")

# Save the normalized data
np.save(path_to_save_normalised_data + "X_train_normalised.npy", X_train_normalised)
np.save(path_to_save_normalised_data + "X_test_normalised.npy", X_test_normalised)
np.save(path_to_save_normalised_data + "X_val_normalised.npy", X_val_normalised)
np.save(path_to_save_normalised_data + "labels_train_normalised.npy", labels_train_normalised)
np.save(path_to_save_normalised_data + "labels_test_normalised.npy", labels_test_normalised)
np.save(path_to_save_normalised_data + "labels_val_normalised.npy", labels_val_normalised)
print(f"- Normalized data and labels saved to {path_to_save_normalised_data}")

np.save(path_to_save_min_max_of_labels + "labels_train_min.npy", min_labels_train)
np.save(path_to_save_min_max_of_labels + "labels_train_max.npy", max_labels_train)
np.save(path_to_save_min_max_of_labels + "labels_test_min.npy", min_labels_test)
np.save(path_to_save_min_max_of_labels + "labels_test_max.npy", max_labels_test)
np.save(path_to_save_min_max_of_labels + "labels_val_min.npy", min_labels_val)
np.save(path_to_save_min_max_of_labels + "labels_val_max.npy", max_labels_val)
print(f"- Min and max of labels saved to {path_to_save_min_max_of_labels}")



print("- All done :)")



