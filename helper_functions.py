import os
import shutil
import random
from glob import glob

import numpy as np
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pathlib import Path
import json
from tqdm import tqdm

import datetime, time
from collections import ChainMap
from distutils.dir_util import copy_tree

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# generate classes     
def generate_classlist_all(dataset_dir):
    '''
    Generates a list of all classes based from the looped data.
    
    Parameters
    ------
    dataset_dir - dataset directory
    '''
    
    class_list_all = glob(os.path.join(dataset_dir, '*',))
    class_list_all = [i.split('\\')[-1] for i in class_list_all]
    return class_list_all

# generate class
def generate_classlist(class_list, num_class):
    '''
    Generates a list of n classes.
    
    Parameters
    ------
    class_list - a list of classes
    num_classes - amount of classes you want
    '''
    counter = 0
    class_list_ = []
    while counter != num_class:
        if np.random.choice(class_list) not in class_list_:        
            class_list_.append(np.random.choice(class_list))
            counter +=1
    return class_list_
    

def generate_dict_data(classlist, dataset_dir):
    '''
    Generates a dictionary of class along with its list of data
    
    Parameters
    ------
    classlist - list of classes
    dataset_dir - dataset directory
    '''
    
    dict_ = {}
    for class_ in tqdm(classlist):
        dict_[class_] = []
    for class_name, list_ in dict_.items():
        for filenames in os.listdir(dataset_dir + class_name):
            list_.append(filenames)
    return dict_

def generate_train_test_data(dict_files, percentage_train):
    '''
    Generates a train-test split with the percentage given.
    
    Parameters
    ------
    dict_files - dictionary of classes along with its list of data
    percent_test - percent of train data to generate
    
    Usage
    ----
    train, test = generate_train_test_data(dict_files, percentage_test)
    '''
    test_data_all = []
    train_data_all = []
    
    percentage = percentage_train / 100
    
    for name_, data_ in tqdm(dict_files.items()):
        test_data_all.append({name_ : dict_files[name_][(int(len(dict_files[name_])*percentage)):]})
        train_data_all.append({name_ : dict_files[name_][:(int(len(dict_files[name_])*percentage))]})
            
    # combines lists into a single dictionary        
    test_data_all = dict(ChainMap(*test_data_all))
    train_data_all = dict(ChainMap(*train_data_all))        
    return train_data_all, test_data_all

def build_train_test_datafiles(classlist, destination_folder, folder_name):
    '''
    takes in a class list, and generates folders locally
    
    Parameters
    -----
    classlist - list of classes
    train_dir - local directory for train files
    test_dir - local directory for test files
    '''
    for i in classlist:
        Path(destination_folder + folder_name + '/' + i).mkdir(parents=True, exist_ok=True),
           
def generate_data_locally(classlist, data, source_folder, destination_folder):
    '''
    Copies and paste data towards the destination folder
    
    Parameters
    ------
    classlist - list of classes
    data - dictionary of train/test data
    source_folder - location of the data
    destination_folder - location you want to paste it
    percentage - amount of percentage within you want to pase
    '''
    
    #percent = percentage/100
    for names_ in tqdm(classlist):
        #for files in data[names_][:(int(len(data[names_]) * percent))]:
        for files in data[names_]:
            # construct full file path
            source = source_folder + names_ + '/' + files
            destination = destination_folder + names_+ '/' + files
            # copy only files
            if os.path.isfile(source):
                shutil.copy(source, destination)
        
def generate_data_locally_percent(classlist, data, source_folder, destination_folder, percentage):
    '''
    Copies and paste data towards the destination folder
    
    Parameters
    ------
    classlist - list of classes
    data - dictionary of train/test data
    source_folder - location of the data
    destination_folder - location you want to paste it
    percentage - amount of percentage within you want to pase
    '''
    
    percent = percentage/100
    for names_ in tqdm(classlist):
        for files in data[names_][:(int(len(data[names_]) * percent))]:
        
            # construct full file path
            source = source_folder + names_ + '/' + files
            destination = destination_folder + names_+ '/' + files
            # copy only files
            if os.path.isfile(source):
                shutil.copy(source, destination)
        

def loop_through_files(dataset_dir):
    '''
    Loops through the directory and prints out the how much data within
    
    Parameters
    ------
    data_dir(str) - dataset directory
    '''
    #total_files = []
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        print(filenames)
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'.")
        #total_files.append(len(filenames))
        #print(sum(total_files))

def plot_random_image(class_list, directory):
    
    '''
    Takes in a list of classes which randomly selects an image in its directory
    
    
    Parameters
    --------
    class_list - list of classes(similar to its folder name)
    directory - location of images you would want to view
    '''
    
    random_class = random.choice(class_list)
    random_dir = directory + random_class
    random_img = random.choice(os.listdir(random_dir))
    rng_dir_img = random_dir + '/' + random_img
    arr_dir_img = mpimg.imread(rng_dir_img)
    
    img=plt.imshow(arr_dir_img)
    plt.axis(False)
    plt.title(f'class:{random_class}\n shape:{arr_dir_img.shape}\n {random_img}')
    return img

def plot_history_curves(model):
    '''
    Plots a model's history metric
    
    Parameters
    ------
    model - model history
    '''
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    
    accuracy = model.history['accuracy']
    val_accuracy = model.history['val_accuracy']
    
    epochs = range(len(model.history['loss']))
    
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss[lower == better]')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.figure()
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Val Accuracy')
    plt.title('Accuracy[higher==better]')
    plt.xlabel('Epochs')
    plt.legend()

def plot_random_image(class_list, directory, titles=True):
    
    '''
    Takes in a list of classes which randomly selects an image in its directory
    and outputs the image file, along with the img directory
    
    Parameters
    --------
    class_list - list of classes(similar to its folder name)
    directory - location of images you would want to view
    '''
    
    random_class = random.choice(class_list)
    random_dir = directory + random_class
    random_img = random.choice(os.listdir(random_dir))
    rng_dir_img = random_dir + '/' + random_img
    arr_dir_img = mpimg.imread(rng_dir_img)
    
    plt.axis(False)
    if titles:
        img=plt.imshow(arr_dir_img)
        plt.title(f'class:{random_class}\n shape:{arr_dir_img.shape}\n {random_img}')
    else:
        img=plt.imshow(arr_dir_img)
        
    return img, rng_dir_img

def plot_compare_history(original_history, new_history, initial_epochs):
    '''
    Input model_2 then new history to show frequency of the train and val loss
    '''
    
    acc = original_history.history['accuracy']
    loss = original_history.history['loss']
    
    val_acc = original_history.history['val_accuracy']
    val_loss = original_history.history['val_loss']
    
    total_acc = acc + new_history.history['accuracy']
    total_loss = loss + new_history.history['loss']
    
    total_val_acc = val_acc + new_history.history['val_accuracy']
    total_val_loss = val_loss + new_history.history['val_loss']
    
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation History')
    
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1], plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

def plot_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
    '''
    Makes a labelled confusion matrix comparing predictions and ground truth labels.

    If classes is passed, confusion matrix will be labelled, if not, integer class values
    will be used.

    Args:
      y_true: Array of truth labels (must be same shape as y_pred).
      y_pred: Array of predicted labels (must be same shape as y_true).
      classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
      figsize: Size of output figure (default=(10, 10)).
      text_size: Size of output figure text (default=15).
      norm: normalize values or not (default=False).
      savefig: save confusion matrix to file (default=False).

    Returns:
      A labelled confusion matrix plot comparing y_true and y_pred.

    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10)
    '''
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
    n_classes = cm.shape[0]  # find the number of classes we're dealing with

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
    fig.colorbar(cax)

    # Are there a list of classes?
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title="Confusion Matrix",
           xlabel="Predicted label",
           ylabel="True label",
           xticks=np.arange(n_classes),  # create enough axis slots for each class
           yticks=np.arange(n_classes),
           xticklabels=labels,  # axes will labeled with class names (if they exist) or ints
           yticklabels=labels)

    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    
    plt.xticks(rotation=90)

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    # Save the figure to the current working directory
    if savefig:
        fig.savefig("confusion_matrix.png")

def create_checkpoint_callback(checkpoint_path, metric, save_best=False, save_weights=False):
    '''
    creates a checkpoint callback
    
    Parameters
    -----
    checkpoint_path - destination file path
    metric - metric you'd want to follow
    save_best, save_weights - bool 
    '''
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=metric,
        save_weights_only=save_weights,
        save_best_only=save_best,
        verbose=1
    )
    return checkpoint_callback

def create_early_stopping_callback(metric, patience):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=metric, 
                                                  patience=patience)
    return early_stopping

def create_reduce_lr_callback(metric, factor, patience, verbose, min_lr):
    '''
    Reduce learning rate when a metric has stopped improving.

    Parameters
    ------
    monitor: quantity to be monitored.
    factor: factor by which the learning rate will be reduced. new_lr = lr * factor.
    patience: number of epochs with no improvement after which learning rate will be reduced.
    verbose: int. 0: quiet, 1: update messages.
    mode: one of {'auto', 'min', 'max'}. In 'min' mode, the learning rate will be reduced when the quantity monitored has stopped decreasing; in 'max' mode it will be reduced when the quantity monitored has stopped increasing; in 'auto' mode, the direction is automatically inferred from the name of the monitored quantity.
    min_delta: threshold for measuring the new optimum, to only focus on significant changes.
    cooldown: number of epochs to wait before resuming normal operation after lr has been reduced.
    min_lr: lower bound on the learning rate.
    '''

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=metric,  
                                                 factor=factor,
                                                 patience=patience,
                                                 verbose=verbose, 
                                                 min_lr=min_lr)
    return reduce_lr

# Building a Tensorboard Function

def create_tensorboard_callback(dir_name, experiment_name):
    '''
    creates a tensorboard callback logs file
    
    dirname / logs/fits / experiment_name / datetime
    
    Parameters
    -------
    dir_name - project path name where logs will be stored
    experiment_name - specific name of the model path you'd want to generate
    
    %load_ext tensorboard
    tensorboard --logdir='' --host localhost
    '''
    log_dir = dir_name + '/' + 'logs/fits' + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f'Saving Tensorboard logfiles to {log_dir}')
    return tensorboard_callback

    
    

def evaluate_model_results(y_preds, y_true):
    '''
    returns a list of model evaluation results
    
    Paremeters
    -------
    y_preds - predictions from the model
    y_true - true labels from test split
    '''
    model_accuracy = accuracy_score(y_true, y_preds)
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_preds, average='weighted')
    
    model_results = {'accuracy' : model_accuracy,
                     'precision' : model_precision,
                     'recall': model_recall,
                     'f1-score': model_f1}
    return model_results

    
def evaluate_text_preds(y_preds, y_true, y_true_sentences):
    '''
    outputs a list of prediction, true labels, and its sentence
    
    Parameters
    -------
    y_preds - predictions from the model
    y_true - true labels from test split
    y_true_sentences - sentence of the true label
    '''
    pred_results = []
    for i in range(0, len(y_preds)):
        if y_preds[i] == y_true[i]:
            #print('correct prediction')
            pred_results.append('Correct Prediction')
        else:
            #print('wrong prediction')
            pred_results.append('Wrong Prediction')
            
    for i in range(0, len(pred_results)):
        print('---\n')
        print(f'{pred_results[i]} \n'
              f'Prediction: {y_preds[i]} | True Label: {y_true[i]}\n\n'
              f'{y_true_sentences[i]}\n')