import numpy as np
import pathlib
from skimage.transform import rescale

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


num_samples = 3000
# Normalize the data
X_BC_train_normalised = X_BC_train[:num_samples].astype(np.float16) / 255.0 
X_M_train_normalised = X_M_train[:num_samples].astype(np.float16) / 255.0
X_BC_test_normalised = X_BC_test[:500].astype(np.float16) / 255.0
X_M_test_normalised = X_M_test[:500].astype(np.float16) / 255.0
X_BC_val_normalised = X_BC_val[:500].astype(np.float16) / 255.0
X_M_val_normalised = X_M_val[:500].astype(np.float16) / 255.0 


# Stack the data because model architecture requires it
X_train_normalised = np.stack((X_BC_train_normalised, X_M_train_normalised), axis=1)
X_test_normalised = np.stack((X_BC_test_normalised, X_M_test_normalised), axis=1)
X_val_normalised = np.stack((X_BC_val_normalised, X_M_val_normalised), axis=1)


def rescale_images(data, scale_factor):
    rescaled_images = np.empty((data.shape[0], data.shape[1], int(scale_factor*data.shape[2]), int(scale_factor*data.shape[3]), data.shape[4]),dtype=np.float16)
    num_samples = data.shape[0]
    for i in range(num_samples):
        for j in range(2):
            # Rescale the image and store it
            rescaled_images[i, j] = rescale(X_train_normalised[i, j], scale_factor, 
                                            anti_aliasing=True, channel_axis=-1)
    return rescaled_images


scale_factor = 0.25
X_train_rescaled = rescale_images(X_train_normalised, scale_factor)
X_test_rescaled = rescale_images(X_test_normalised, scale_factor)
X_val_rescaled = rescale_images(X_val_normalised, scale_factor)

np.save(path_to_save_normalised_data + "X_train_rescaled.npy", X_train_rescaled)
np.save(path_to_save_normalised_data + "X_test_rescaled.npy", X_test_rescaled)
np.save(path_to_save_normalised_data + "X_val_rescaled.npy", X_val_rescaled)

