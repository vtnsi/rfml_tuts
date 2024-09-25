# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:51:22 2024

script for generating test data for the harness notebook

@author: scadams21
"""

import torch
from utils.data_utils import IQ_data_gen


# signal list
signal_list = [{"format":"ask", "order":16, "label": "16ASK"},
               {"format":"pam", "order":16, "label": "16PAM"},
               {"format":"psk", "order":16, "label": "16PSK"},
               {"format":"qam", "order":16, "label": "16QAM"}]
num_classes = len(signal_list)

# signal config file locations
signal_filenames = ['configs/' + signal["label"] + '.json' for signal in signal_list]

# number of sequences in the test set and length of each sequence
num_seq = 250
seq_len = 256

data, labels, label_dict = IQ_data_gen(signal_filenames, num_seq, seq_len)

# convert labels to array
y = torch.zeros((num_seq*num_classes, num_classes))
for idx, label in enumerate(labels):
    y[idx, :] = torch.tensor(label_dict[label])

dataset = [torch.transpose(data, 1, 2), y]
torch.save(dataset, 'resources/test_dataset.pt')


