# ----------------------------------------------------------------------------
# Copyright (c) 2017-2019, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from sklearn.metrics import (
    mean_squared_error, confusion_matrix, accuracy_score)

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import linregress
import matplotlib.pyplot as plt


def _custom_palettes():
    return {
        'YellowOrangeBrown': 'YlOrBr',
        'YellowOrangeRed': 'YlOrRd',
        'OrangeRed': 'OrRd',
        'PurpleRed': 'PuRd',
        'RedPurple': 'RdPu',
        'BluePurple': 'BuPu',
        'GreenBlue': 'GnBu',
        'PurpleBlue': 'PuBu',
        'YellowGreen': 'YlGn',
        'summer': 'summer_r',
        'copper': 'copper_r',
        'viridis': 'viridis_r',
        'plasma': 'plasma_r',
        'inferno': 'inferno_r',
        'magma': 'magma_r',
        'sirocco': sns.cubehelix_palette(
            dark=0.15, light=0.95, as_cmap=True),
        'drifting': sns.cubehelix_palette(
            start=5, rot=0.4, hue=0.8, as_cmap=True),
        'melancholy': sns.cubehelix_palette(
            start=25, rot=0.4, hue=0.8, as_cmap=True),
        'enigma': sns.cubehelix_palette(
            start=2, rot=0.6, gamma=2.0, hue=0.7, dark=0.45, as_cmap=True),
        'eros': sns.cubehelix_palette(start=0, rot=0.4, gamma=2.0, hue=2,
                                      light=0.95, dark=0.5, as_cmap=True),
        'spectre': sns.cubehelix_palette(
            start=1.2, rot=0.4, gamma=2.0, hue=1, dark=0.4, as_cmap=True),
        'ambition': sns.cubehelix_palette(start=2, rot=0.9, gamma=3.0, hue=2,
                                          light=0.9, dark=0.5, as_cmap=True),
        'mysteriousstains': sns.light_palette(
            'baby shit green', input='xkcd', as_cmap=True),
        'daydream': sns.blend_palette(
            ['egg shell', 'dandelion'], input='xkcd', as_cmap=True),
        'solano': sns.blend_palette(
            ['pale gold', 'burnt umber'], input='xkcd', as_cmap=True),
        'navarro': sns.blend_palette(
            ['pale gold', 'sienna', 'pine green'], input='xkcd', as_cmap=True),
        'dandelions': sns.blend_palette(
            ['sage', 'dandelion'], input='xkcd', as_cmap=True),
        'deepblue': sns.blend_palette(
            ['really light blue', 'petrol'], input='xkcd', as_cmap=True),
        'verve': sns.cubehelix_palette(
            start=1.4, rot=0.8, gamma=2.0, hue=1.5, dark=0.4, as_cmap=True),
        'greyscale': sns.blend_palette(
            ['light grey', 'dark grey'], input='xkcd', as_cmap=True)}


def _regplot_from_dataframe(x, y, plot_style="whitegrid", arb=True,
                            color="grey"):
    '''Seaborn regplot with true 1:1 ratio set by arb (bool).'''
    sns.set_style(plot_style)
    reg = sns.regplot(x, y, color=color)
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    if arb is True:
        x0, x1 = reg.axes.get_xlim()
        y0, y1 = reg.axes.get_ylim()
        lims = [min(x0, y0), max(x1, y1)]
        reg.axes.plot(lims, lims, ':k')
    return reg


def _linear_regress(actual, pred):
    '''Calculate linear regression on predicted versus expected values.
    actual: pandas.DataFrame
        Actual y-values for test samples.
    pred: pandas.DataFrame
        Predicted y-values for test samples.
    '''
    slope, intercept, r_value, p_value, std_err = linregress(actual, pred)
    mse = mean_squared_error(actual, pred)
    return pd.DataFrame(
        [(mse, r_value, r_value**2, p_value, std_err, slope, intercept)],
        columns=["Mean squared error", "r-value", "r-squared", "P-value",
                 "Std Error", "Slope", "Intercept"],
        index=[actual.name])


def _plot_heatmap_from_confusion_matrix(cm, palette):
    palette = _custom_palettes()[palette]
    return sns.heatmap(cm, cmap=palette, cbar_kws={'label': 'Proportion'})


def _add_sample_size_to_xtick_labels(ser, classes):
    '''ser is a pandas series.'''
    labels = ['{0} (n={1})'.format(c, ser[ser == c].count()) for c in classes]
    return labels


def _plot_confusion_matrix(y_test, y_pred, classes, normalize, palette):

    accuracy = accuracy_score(y_test, pd.DataFrame(y_pred))
    cm = confusion_matrix(y_test, y_pred)
    # normalize
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # fill na values (e.g., true values that were not predicted) otherwise
    # these will appear as whitespace in plots and results table.
    cm = np.nan_to_num(cm)

    confusion = _plot_heatmap_from_confusion_matrix(cm, palette)

    x_tick_labels = _add_sample_size_to_xtick_labels(y_pred, classes)
    y_tick_labels = _add_sample_size_to_xtick_labels(y_test, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    confusion.set_xticklabels(x_tick_labels, rotation=90, ha='center')
    confusion.set_yticklabels(y_tick_labels, rotation=0, ha='right')

    # generate confusion matrix as pd.DataFrame for viewing
    predictions = pd.DataFrame(cm, index=classes, columns=classes)
    # add empty row/column to show overall accuracy in bottom right cell
    # baseline error = error rate for a classifier that always guesses the
    # most common class
    n_samples, n_samples_largest_class, basline_accuracy, accuracy_ratio = \
        _calculate_baseline_accuracy(y_test, accuracy)
    predictions["Overall Accuracy"] = ""
    predictions.loc["Overall Accuracy"] = ""
    predictions.loc["Baseline Accuracy"] = ""
    predictions.loc["Accuracy Ratio"] = ""
    predictions.loc["Overall Accuracy"]["Overall Accuracy"] = accuracy
    predictions.loc["Baseline Accuracy"]["Overall Accuracy"] = basline_accuracy
    predictions.loc["Accuracy Ratio"]["Overall Accuracy"] = accuracy_ratio

    return predictions, confusion


def _calculate_baseline_accuracy(y_test, accuracy):
    n_samples = len(y_test)
    n_samples_largest_class = y_test.value_counts().iloc[0]
    basline_accuracy = n_samples_largest_class / n_samples
    accuracy_ratio = accuracy / basline_accuracy
    return n_samples, n_samples_largest_class, basline_accuracy, accuracy_ratio


def _plot_RFE(x, y):
    rfe = plt.figure()
    plt.xlabel("Feature Count")
    plt.ylabel("Accuracy")
    plt.plot(x, y, 'grey')
    return rfe
