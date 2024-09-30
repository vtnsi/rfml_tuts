# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 11:28:08 2023

data utils new

uses py_waspgen data generation

@author: scadams21
"""

import json
import numpy as np
import pandas as pd


import torch
from torch.utils.data import Dataset

# Py-waspgen packages
import pywaspgen.burst_datagen as burst_datagen
import pywaspgen.iq_datagen as iq_datagen

from .featuredata_gen import featuredata_gen


class IQ_Dataset(Dataset):
    '''
    Class for custom pytorch dataset for IQ data
    '''

    # def __init__(self, data, labels, label_dict, CNN_Data=False, transpose = False, ensemble=False):
    def __init__(self, data, labels, label_dict):
        '''
        Initialize IQ_Dataset class

        Args:
            data: Torch tensor of data from IQ_data_gen function
                dimensions [num_sequences, seq_len, 2]
            labels: List of labels from IQ_data_gen
            label_dict: Dictionary that converts label to tensor
        '''
        self.data = data
        self.labels = labels
        self.label_dict = label_dict
        # self.CNN_Data = CNN_Data
        # self.transpose = transpose
        # self.ensemble = ensemble

    def __len__(self):
        '''
        Returns the number of examples in the data set
        '''
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Returns the X and y for idx
        '''
        return torch.transpose(self.data[idx], 0, 1), self.label_dict[self.labels[idx]]


def normalize_fun(s):
    '''
    Function to normalize an IQ signal

    Args:
        s: IQ signal to be normalized
            dimensions [obs_len, 2]

    Returns:
        s_norm: Normalized IQ signal
    '''
    pwr = np.sqrt(np.mean(np.power(np.abs(s), 2.)))
    return s/pwr


def IQ_data_gen(sig_filenames, num_seq, obs_len, generate_features=False):
    '''
    Wrapper function for generating data
    Args:
          sig_filenames: list of configuration files for each modulation type
          num_est: number of sequences to generation
          obs_len: sequence length
          generate_features: indicator to generate features from the signal
              default: False

    Returns:
          data: tensor of data [NxLxF]
          labels: tensor of labels
          label_dict: dictionary that converts label to a tensor
    '''

    # initialize objects to store information
    data = torch.empty((len(sig_filenames)*num_seq, obs_len, 2), dtype=torch.float)
    labels = []
    label_dict = {}
    num_classes = len(sig_filenames)

    if generate_features:
        sigma_aa, sigma_dp, sigma_ap, sigma_af, kurtosis_a, kurtosis_f, symmetry = [], [], [], [], [], [], []

    count = 0
    # loop over the signal type
    for idx, sig_filename in enumerate(sig_filenames):

        # load config file
        config_file_id = open(sig_filename)
        configs = json.load(config_file_id)

        # create label_dict entry for the signal type
        target = np.zeros(num_classes)
        target[idx] = 1
        label = configs["spectrum"]["sig_types"][0]["label"]
        label_dict[label] = target

        # loop over the number of sequences for the signal
        for k in range(num_seq):

            # create the burst list from py_waspgen
            burst = burst_datagen.BurstDatagen(sig_filename)
            burst_list = burst.gen_burstlist()

            # generate the data using py_waspgen
            iq_gen = iq_datagen.IQDatagen(sig_filename)
            iq_data, _ = iq_gen.gen_iqdata(burst_list)

            # normalize data
            iq_data = normalize_fun(iq_data)

            # generate features if functionality is turned on
            if generate_features:
                fg = featuredata_gen(iq_data)
                sigma_aa.append(fg.sigma_aa())
                sigma_dp.append(fg.sigma_dp())
                sigma_ap.append(fg.sigma_ap())
                sigma_af.append(fg.sigma_af())
                kurtosis_a.append(fg.kurtosis_a())
                kurtosis_f.append(fg.kurtosis_f())
                symmetry.append(fg.symmetry())

            # convert to tensors and store
            data[count, :, 0] = torch.from_numpy(iq_data.real).to(torch.float)
            data[count, :, 1] = torch.from_numpy(iq_data.imag).to(torch.float)
            labels.append(label)

            count += 1

    if generate_features:
        # create the feature data frame
        feature_dict = {'sigma_aa': sigma_aa, 'sigma_dp': sigma_dp,
                        'sigma_ap': sigma_ap, 'sigma_af': sigma_af,
                        'kurtosis_a': kurtosis_a, 'kurtosis_f': kurtosis_f,
                        'symmetry': symmetry}
        feature_df = pd.DataFrame(feature_dict)
        return data, labels, label_dict, feature_df
    else:
        return data, labels, label_dict


def create_signal_jsons(dir_name,
                        signal_list,
                        observation_duration=256,
                        cent_freq=[0.0, 0.0],
                        bandwidth=[0.5, 0.5],
                        start=[0, 0],
                        duration=[256, 256],
                        snr=[5, 5]):

    '''
    function for creating signal json files with common parameters

    Args:
        dir_name: name of directory for storing the config json files
        signal_list: list of signals to create config files for
        observation_duration: total length of the generatd data.  Default is
        256.
        cent_freq: center frequency for the signal.  Default is [0.0, 0.0].
            If the elements of the list are different, py_waspgen will randomly
            sample the center frequency over the interval.
        bandwidth: bandwidht of the signal.  Default is [0.5, 0.5].  If
            the elements of the list are different, py_waspgen will randomly
            sample the bandwidth over the interval.
        start: start time for the signal.  Default is [0, 0].  If
            the elements of the list are different, py_waspgen will randomly
            sample the start time over the interval.
        duration: duration of the signal.  Default is [256, 256].  If
            the elements of the list are different, py_waspgen will randomly
            sample the duration over the interval.
        snr: signal to noise ratio.  Default is [5, 5].  If
            the elements of the list are different, py_waspgen will randomly
            sample the snr over the interval.
    '''

    for signal in signal_list:

        json_dict = {"generation": {"rand_seed": 42, "pool": 1},
                     "spectrum": {"observation_duration": observation_duration,
                                  "sig_types": [signal],
                                  "max_signals": 1,
                                  "allow_collisions_flag": False,
                                  "max_attempts": 100,
                                  "save_modems": False},
                     "burst_defaults": {"cent_freq": cent_freq,
                                        "bandwidth": bandwidth,
                                        "start": start,
                                        "duration": duration},
                     "iq_defaults": {"snr": snr},
                     "pulse_shape_defaults": {"format": "RRC",
                                              "beta": [ 0.1, 0.9 ],
                                              "span": [ 10, 20 ],
                                              "window": {
                                                  "type": "kaiser",
                                                  "params": 5}}
                     }

        filename = dir_name + '/' + signal["label"] + '.json'
        with open(filename, "w") as file:
            json.dump(json_dict, file, indent=6)
