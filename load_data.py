import numpy as np
import mne
from ann_preprocessing import create_feature_vector
import os


def load_info(info_path):
    """
    function that loads then information saved in a txt file for each proband
    input: path to the file
    output: one array where the info for seizures in saved and the other where the info for noneseizures is saved
        the seizure array contains strings of file name, start and stop
        the non_seizure contains strings of file name
    """
    with open(info_path, 'r') as infile:
        info_string = infile.read()

    info_array = info_string.split('\n')

    seizure_info = []
    non_seizure_info = []

    for i in range(len(info_array)):
        # until end of file is not reached
        if i + 3 <= len(info_array) - 1:
            # check if seizure
            if 'File Name' in info_array[i]:

                # check for all the seizures
                # Max amount of seizure in one file is 3
                if 'File: 3' in info_array[i + 3]:
                    seizure_info.append(info_array[i:i + 6])
                    seizure_info.append([info_array[i], info_array[i + 6], info_array[i + 7]])
                    seizure_info.append([info_array[i], info_array[i + 8], info_array[i + 9]])

                # in case we have two seizures in one file
                if 'File: 2' in info_array[i + 3]:
                    seizure_info.append(info_array[i:i + 6])
                    seizure_info.append([info_array[i], info_array[i + 6], info_array[i + 7]])
                    i = i + 7

                # in case we have one seizure in File
                if 'File: 1' in info_array[i + 3]:
                    seizure_info.append(info_array[i:i + 6])
                    i = i + 5

                # If we don't have any seizure, we still want the name of the File
                else:
                    non_seizure_info.append(info_array[i])
                    i = i + 3

    return seizure_info, non_seizure_info


def create_path(prob_nr, small):
    """
    Creates a path to the folder, generated from the number of the proband
    input: the number of proband and if it is smaller than 10
    output: complete path as string
    """
    # small indicated if the number is < 10
    return 'D:\Bachelor_Arbeit\Data\MIT-CHB\p_' + prob_nr + '\\'
    # if small:
    #     return 'D:\Bachelor_Arbeit\Data\MIT-CHB\p_'+ prob_nr + '\\'
    # #this seems to be redundand.. check again
    # else:
    #     return 'D:\Bachelor_Arbeit\Data\MIT-CHB\p_'+ prob_nr + '\\'


def load_data(path, start, stop):
    """
    Uses MNE-toolbox to load the data. Apply noth filter at 50 Hz
    Input: path to file, start and stop in seconds as float
    output: the MNE.io.Raw object that contains the given time intervall
    """
    # check if path exist and then load the object with mne
    if os.path.isfile(path):

        raw = mne.io.read_raw_edf(path, exclude=['-', 'T8-P8', '.'], verbose=False, preload=True)

        # apply notch filter around 50 Hz.
        raw.notch_filter(freqs=[49.1, 50.9])

        # Stim channel is automatically created by MNE.
        raw.drop_channels(['STI 014'])

        # Take 30 seconds before onset
        start = start - 30
        stop = start + 150

        # check if the recording is long enough to create the feature vector
        if raw.n_times // 256 > start + 150:
            try:
                cropped = raw.crop(start, stop, verbose=False)
            except:
                # Sometime something goes wrong so I manually extract the seizure of the timewindow and return a
                # newly created object with the info
                manual_raw = raw.get_data()
                freq = raw.info['sfreq']
                manual_start = int(start * freq)
                manual_stop = int((stop * freq + 1))
                manual_data = np.asarray([manual_raw[i][manual_start:manual_stop] for i in range(len(manual_raw))])
                cropped = mne.io.RawArray(manual_data, raw.info, verbose=False)

        return cropped


def get_seizures(seizure_info, small):
    """
    Loads the seizures from the summary.txt file in the folder of each subject.
    input:  seizure_info is an array created by the load_info function, which contains the file number and start and
            stop of the seizures.
            small: bool of the number is smaller than 10
    ouput: array of MNE.io.Raw objects containing all the seizures
    """
    seizures = []

    # Use file_info to get the proband number and create a general path to the prob number
    file = seizure_info[0][0].split(':')[1].split(' ')[1]
    proband_nr = file[3:5]
    gen_path = create_path(proband_nr, small)

    # get the time windows of every seizure of the above prob
    # and load the data as a mne.io.Raw object
    for seizure in seizure_info:
        # extract start and stop from the summary txt file
        start = int(seizure[-2].split(':')[1].split(' ')[1])
        stop = int(seizure[-1].split(':')[1].split(' ')[1])
        f = seizure[0].split(':')[1].split(' ')[1]
        file_nr = f[6:8]
        seiz_path = gen_path + 'chb' + proband_nr + '_' + file_nr + '.edf'

        # load the mne object with the iven path
        seizures.append(load_data(seiz_path, start, stop))

    # assert np.asarray(seizures).all() is not None
    return seizures


def get_non_seizure(info, small):
    """
    input: non_seizure_info array containing all the files without seizures
            bool if the prob number is smaller than 10
    """
    non_seizures = []
    file = info[0].split(':')[1].split(' ')[1]
    proband_nr = file[3:5]
    gen_path = create_path(proband_nr, small)

    for recording in info:

        f = recording.split(':')[1].split(' ')[1]
        file_nr = f[6:8]
        non_seiz_path = gen_path + 'chb' + proband_nr + '_' + file_nr + '.edf'
        if os.path.isfile(non_seiz_path):
            raw = mne.io.read_raw_edf(non_seiz_path, preload=True, exclude=['-', 'T8-P8', '.'], verbose=False)

            # apply notch filter and drop the automatically created stim channel
            raw.notch_filter(freqs=[49.1, 50.9])
            raw.drop_channels(['STI 014'])

            # calculate the length of the intervall where we select random sniplets from
            overall_length = (raw.n_times // 256) - 1
            intervall_length = overall_length // 3

            # make sure we don't run into problems with the intervall length
            if (3 * intervall_length) < (raw.n_times // 256):
                start_stop_array = get_intervalls(intervall_length)

                # take three different time windows for the non seizure vector
                for i in range(3):
                    try:
                        cropped = raw.crop(start_stop_array[i], (start_stop_array[i] + 150), verbose=False)
                    except:
                        # in case something goes wrong Manual moe takes over
                        manual_raw = raw.get_data()
                        freq = raw.info['sfreq']
                        manual_start = int(start_stop_array[i] * freq)
                        manual_stop = int((start_stop_array[i] + 150) * freq + 1)
                        manual_data = np.asarray(
                            [manual_raw[i][manual_start:manual_stop] for i in range(len(manual_raw))])
                        cropped = mne.io.RawArray(manual_data, raw.info, verbose=False)

                    non_seizures.append(cropped)

    return non_seizures


def get_intervalls(intervall_length):
    """
    Creates random time windows in pre defined sections of the recording
    input: the length of the intervall
    output: an array of three ints, where the cropping starts
    """
    intervalls = []

    intervalls.append(np.random.randint(40, intervall_length - 155))
    intervalls.append(np.random.randint(intervall_length, 2 * intervall_length - 155))
    intervalls.append(np.random.randint(2 * intervall_length, 3 * intervall_length - 155))

    return intervalls


def get_ann_features(seiz_info, non_seiz_info, small, all_bands):
    """
    uses the function create_feature_vector from the preprocessing script
    to create a n dimensional feature vector to feed into the ann,svm and rdf

    input: seiz_info - extracted information from the summary txt
           non_seiz_info - extracted info from the summary txt
           small - boolean if the number of the path is smaller than 10
           all_bands - if True all bands are used. else only theta and alpha
    """

    seiz = get_seizures(seiz_info, small)
    non_seiz = get_non_seizure(non_seiz_info, small)

    seiz_vector = []
    non_seiz_vector = []

    for s in seiz:
        seiz_vector.append(create_feature_vector(s, all_bands))

    for ns in non_seiz:
        non_seiz_vector.append(create_feature_vector(ns, all_bands))

    return np.asarray(seiz_vector), np.asarray(non_seiz_vector)


def load_all_feature_vectors(all_bands=True):
    """
    Loads all the seizures and non_seizure
    Input: None
    Output: Seizures array, containing all MNE.io.Raw structures with seizures
           Non_seizures array, containing all MNE.io.Raw structures without seizures, but with the same time window
    """

    ann_seizure_features = []
    ann_non_seizure_features = []

    for i in range(1, 20):
        # Exclude all the probs that are under 4 years old
        if i == 6 or i == 8 or i == 10 or i == 12 or i == 13 or i == 15:
            pass
        else:
            # a 0 has to ba added to the path
            if i < 10:
                p = 'D:\Bachelor_Arbeit\Data\MIT-CHB\p_0' + str(i) + '\\' + 'chb0' + str(i) + '-summary.txt'
                small = True
            else:
                p = 'D:\Bachelor_Arbeit\Data\MIT-CHB\p_' + str(i) + '\\' + 'chb' + str(i) + '-summary.txt'
                small = False

            # load the info
            seizure_info, non_seizure_info = load_info(p)

            # get the frature vectors for the ann only
            seiz, non_seiz = get_ann_features(seizure_info, non_seizure_info, small, all_bands)
            for ind in range(len(seiz)):
                ann_seizure_features.append(seiz[ind])
            for non_s_ind in range(len(non_seiz)):
                ann_non_seizure_features.append(non_seiz[non_s_ind])
            print("Done with ", i)
    return np.asarray(ann_seizure_features), np.asarray(ann_non_seizure_features)

# load_all_feature_vectors(svm = False, all_bands= True)
