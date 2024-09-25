# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 09:05:12 2023

training utilities

@author: scadams21
"""

import numpy as np
import torch


def train_func(model,
               optimizer,
               loss_fn,
               train_dataloader,
               val_dataloader,
               epochs=100,
               print_every_n=1,
               scheduler=None):

    '''
    Function for training a neural network

    Args:
        model: Pytorch model
        optimizer: Pytorch optimizer
        loss_fn: loss function
        train_dataloader: Pytorch data loader for training
        val_dataloader: Pytorch data laoder for validation
        epocs: number of training epochs.  Default set to 100.
        print_every_n: print loss every n iterations.  Default set to 1.
        scheduler: scheduler for the learning rate.  Default is None.

    Returns:
        model: the trained Pytorch model
        train_loss: list of training loss over the epochs
        val_loss: list of validation loss over the epochs
    '''

    # check if a gpu is available for training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == 'cuda':
        model = model.cuda()

    # initialize lists for training and validation loss
    train_loss = []
    val_loss = []

    # training loop
    for epoch in range(epochs):

        if epoch % print_every_n == 0:
            print('Epoch ', epoch+1)

        model.train(True)

        # train
        running_loss = 0
        for X, y in train_dataloader:

            if device == 'cuda':
                X, y = X.cuda(), y.cuda()

            optimizer.zero_grad()
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss += loss

        if scheduler:
            scheduler.step()

        # print epoch training loss
        if epoch % print_every_n == 0:
            print('Training loss: {}'.format(running_loss))

        train_loss.append(running_loss.detach().cpu().numpy())

        # validate
        run_val_loss = 0
        for X, y in val_dataloader:

            if device == 'cuda':
                X, y = X.cuda(), y.cuda()

            y_val = model(X)
            loss = loss_fn(y_val, y)
            run_val_loss += loss

        # print epoch validation loss
        if epoch % print_every_n == 0:
            print('Validation loss: {}'.format(run_val_loss))

        val_loss.append(run_val_loss.detach().cpu().numpy())

    return model, train_loss, val_loss


def predict_func(model, dataloader):
    '''
    function for extracting test and target values

    Args:
        model: pytorch model
        dataloader: pytorch data loader

    Returns:
        y_array: true target values
        y_pred_array: predicted target values
    '''

    device = "cuda" if torch.cuda.is_available() else "cpu"

    y_array = []
    y_pred_array = []
    for X, y in dataloader:

        if device == 'cuda':
            X = X.cuda()

        _, y_target = torch.max(y, dim=1)
        y_array.append(y_target.numpy())
        output = model(X)
        _, y_pred = torch.max(output, dim=1)
        y_pred_array.append(y_pred.cpu())

    return np.concatenate(y_array), np.concatenate(y_pred_array)
