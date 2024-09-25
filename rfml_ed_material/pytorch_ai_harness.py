# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:09:35 2024

AI test harness for pytorch models

@author: scadams21
"""


import torch


class AI_Test_Harness():
    '''
    Pytorch AI test harness class
    
    This is the basic structure for an AI test harness and is written specifically
    for a pytorch model.  This harness construct could be expanded to traditional
    classifiers, such as sci-kit learn, or custom models.
    
    '''
    
    def __init__(self):
        '''
        Initialize the harness class
        '''
        # initialize states for for model and dataset loading 
        self.model_loaded = False
        self.dataset_loaded = False

    
    # method for loading a model
    def load_model(self, model_params_path, model_architecture=None):
        ''' 
        Method for loading a pytorch model.
        
        Args:
            model_params_path (string): path to model parameter file
            model_architecture (object): pytorch class for model
        '''

        self._load_pytorch_model(model_architecture, model_params_path)
        self.model_loaded = True
        
        
    # internal method for loading a pytorch model
    def _load_pytorch_model(self, model_architecture, model_params_path):
        ''' 
        Internal function for loading pytorch model.  Other internal functions
        could be added a later date to expand to other types of model.
        
        Args:
            model_params_path (string): path to model parameter file
            model_architecture (object): pytorch class for model
        '''
        self.model_params_path = model_params_path
        self.model = model_architecture
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(self.model_params_path))
        else:
            self.model.load_state_dict(torch.load(self.model_params_path, map_location=torch.device('cpu')))


    # method for loading a test dataset
    def load_dataset(self, dataset_path):
        '''
        Method for loading a pytorch dataset.
        
        Args:
            dataset_path (string): path to data set
        '''
        dataset = torch.load(dataset_path)
        self.X = dataset[0]
        self.y = dataset[1]
        self.dataset_loaded = True
        
        
    def evaluate(self, metric):
        ''' 
        Method for evaluating the model.  This method works for any type of 
        metric function where the input is metric(y_true, y_pred)
        
        Args:
            metric (function): metric function
            
        Returns:
            Output of metric
        '''
        
        # test to make sure model is loaded
        if not self.model_loaded:
            print('Load model before evaluation')
        # test if dataset is loaded
        elif not self.dataset_loaded:
            print('Load test dataset before evaluation')
        else:
            model_output = self.model(self.X)
            _, y_pred = torch.max(model_output, dim=1)
            _, y_target = torch.max(self.y, dim=1)
            return metric(y_target.detach().numpy(), y_pred.detach().numpy())
