import mne
import numpy as np
import pywt

# 'Delta','Theta','Alpha','Beta','Gamma'
def select_bands(band_array, freqs):
    selected = []
    for band in freqs:
        try:
            if band == 'Delta':
                selected.append(band_array[0])
            elif band == 'Theta':
                selected.append(band_array[1])
            elif band == 'Alpha':
                selected.append(band_array[2])
            elif band == 'Beta':
                selected.append(band_array[3])
            elif band == 'Gamma':
                selected.append(band_array[4])
        except:
            print('Mistake', band)
            pass


    return selected


def get_bands(cropped):
    filter_iter = [
        ('Delta', 0, 4),
        ('Theta', 5, 7),
        ('Alpha', 8, 12),
        ('Beta', 13, 25),
        ('Gamma', 30, 45)]

    if cropped is not None:
        bands = []
        for band, low_freq, high_freq in filter_iter:
            bands.append(cropped.copy().filter(low_freq,
                           high_freq,
                          n_jobs = 1,
                          l_trans_bandwidth = 1,
                          h_trans_bandwidth = 1,
                          verbose=False))

        return bands


def get_selected_data(array):
    data= []
    for mne_object in array:
        data.append(mne_object.get_data())
    return np.asarray(data)


def create_feature_vector(mne_object, all_bands):
    bands = get_bands(mne_object)
    feature_vector = []
    if all_bands == True:
        selected = select_bands(bands, ['Delta','Theta','Alpha','Beta','Gamma'])
        data = get_selected_data(selected)
        feature_vector = []
        try:
            for i in range(5):
                feature_vector.append(np.mean(data[i]))
                feature_vector.append(np.median(data[i]))
                feature_vector.append(np.var(data[i]))
        except:
            pass

    else:
        selected = select_bands(bands, ['Theta','Alpha'])
        data = get_selected_data(selected)

        try:
            feature_vector.append(np.mean(data[0]))
            feature_vector.append(np.median(data[0]))
            feature_vector.append(np.var(data[0]))
            feature_vector.append(np.mean(data[1]))
            feature_vector.append(np.median(data[1]))
            feature_vector.append(np.var(data[1]))
        except:
            pass


    #print(feature_vector)
    return np.asarray(feature_vector)
