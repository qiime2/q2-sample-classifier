# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from sklearn.metrics import (
    mean_squared_error, confusion_matrix, accuracy_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from numpy import interp
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
        'cividis': 'cividis_r',
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
    reg = sns.regplot(x=x, y=y, color=color)
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


def _plot_heatmap_from_confusion_matrix(cm, palette, vmin=None, vmax=None):
    palette = _custom_palettes()[palette]
    plt.figure()
    scaler, labelsize, dpi, cbar_min = 20, 8, 100, .15
    sns.set(rc={'xtick.labelsize': labelsize, 'ytick.labelsize': labelsize,
            'figure.dpi': dpi})
    fig, (ax, cax) = plt.subplots(ncols=2, constrained_layout=True)
    heatmap = sns.heatmap(cm, vmin=vmin, vmax=vmax, cmap=palette, ax=ax,
                          cbar_ax=cax, cbar_kws={'label': 'Proportion'},
                          square=True, xticklabels=True, yticklabels=True)

    # Resize the plot dynamically based on number of classes
    hm_pos = ax.get_position()
    scale = len(cm) / scaler
    # prevent cbar from getting unreadably small
    cbar_height = max(cbar_min, scale)
    ax.set_position([hm_pos.x0, hm_pos.y0, scale, scale])
    cax.set_position([hm_pos.x0 + scale * .95, hm_pos.y0, scale / len(cm),
                     cbar_height])

    # Make the heatmap subplot (not the colorbar) the active axis object so
    # labels apply correctly on return
    plt.sca(ax)
    return heatmap


def _add_sample_size_to_xtick_labels(ser, classes):
    '''ser is a pandas series.'''
    labels = ['{0} (n={1})'.format(c, ser[ser == c].count()) for c in classes]
    return labels


def _plot_confusion_matrix(y_test, y_pred, classes, normalize, palette,
                           vmin=None, vmax=None):

    accuracy = accuracy_score(y_test, pd.DataFrame(y_pred))
    cm = confusion_matrix(y_test, y_pred)
    # normalize
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # fill na values (e.g., true values that were not predicted) otherwise
    # these will appear as whitespace in plots and results table.
    cm = np.nan_to_num(cm)
    _check_vmin_and_vmax(cm, vmin, vmax)

    confusion = _plot_heatmap_from_confusion_matrix(cm, palette, vmin=vmin,
                                                    vmax=vmax)

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


def _check_vmin_and_vmax(cm, vmin, vmax):
    lowest_frequency = np.amin(cm)
    highest_frequency = np.amax(cm)

    error = ''
    if vmin is not None:
        if vmin > lowest_frequency:
            error += ('vmin must be less than or equal to the lowest '
                      'predicted class frequency:\n'
                      f'\t{vmin!r} is greater than {lowest_frequency!r}')
    if vmax is not None:
        if vmax < highest_frequency:
            if error:
                error += '\n'
            error += ('vmax must be greater than or equal to the highest '
                      'predicted class frequency:\n'
                      f'\t{vmax!r} is less than {highest_frequency!r}')
    if error:
        raise ValueError(error)


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


def _binarize_labels(metadata, classes):
    binarized_targets = label_binarize(metadata, classes=classes)
    # to generalize downstream steps, we need to coerce binary data into an
    # array of shape [n_samples, n_classes]
    if len(classes) == 2:
        binarized_targets = np.hstack((
            1 - binarized_targets, binarized_targets))
    return binarized_targets


def _generate_roc_plots(metadata, probabilities, palette):
    '''
    metadata: pd.Series of target values.
    probabilities: pd.DataFrame of class probabilities.
    palette: str specifying sample-classifier colormap name.

    Returns a pretty Receiver Operating Characteristic plot with AUC scores.
    '''
    classes = probabilities.columns
    probabilities = probabilities.values

    # only accepts binary inputs, so binarize the target data
    binarized_targets = _binarize_labels(metadata, classes)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, roc_auc = _roc_per_class(
        binarized_targets, probabilities, classes)

    # Compute micro-average ROC curve and ROC area under curve
    fpr, tpr, roc_auc = _roc_micro_average(
        binarized_targets, probabilities, fpr, tpr, roc_auc)

    # Compute macro-average ROC curve and ROC area
    fpr, tpr, roc_auc = _roc_macro_average(fpr, tpr, roc_auc, classes)

    # generate ROC plot
    colors = _roc_palette(palette, len(classes))
    return _roc_plot(fpr, tpr, roc_auc, classes, colors)


def _roc_palette(palette, n_classes):
    '''
    palette: str specifying sample-classifier colormap name.
    n_classes: int specifying number of classes (== n of colors to select).

    Returns an iterator of colors.
    '''
    palette = _custom_palettes()[palette]

    # specify color palette. Use different specification for str palette name
    # vs. ListedColormap.
    try:
        colors = cycle(sns.color_palette(palette, n_colors=n_classes))
    except TypeError:
        # if using a continuous ListedColormap, select from normalized
        # colorspace. We use linspace start=0.1 to avoid light colors at start
        # of some colormaps.
        palette = palette(np.linspace(0.1, 1, n_classes))
        colors = cycle(palette)
    return colors


# adapted from scikit-learn examples
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def _roc_per_class(binarized_targets, probabilities, classes):
    '''
    binarized_targets: array of binarized class labels of dimensions [n, c],
        where n = number of samples, c = number of classes.
    probabilities: array of class probabilities of dimensions [n, c],
        where n = number of samples, c = number of classes.
    classes: list of classes.

    Returns dicts of False Positive Rate (fpr), True Detection Rate (tdr), and
        ROC Area Under Curve (roc_auc) for each class.
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i, c in zip(range(len(classes)), classes):
        fpr[c], tpr[c], _ = roc_curve(
            binarized_targets[:, i], probabilities[:, i])
        roc_auc[c] = auc(fpr[c], tpr[c])
    return fpr, tpr, roc_auc


# adapted from scikit-learn examples
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def _roc_micro_average(binarized_targets, probabilities, fpr, tpr, roc_auc):
    '''
    binarized_targets: array of binarized class labels of dimensions [n, c],
        where n = number of samples, c = number of classes.
    probabilities: array of class probabilities of dimensions [n, c],
        where n = number of samples, c = number of classes.
    fpr: dict of false-positive rates for each class.
    tdr: dict of true-detection rates for each class.
    roc_auc: dict of auc scores for each class.

    Returns fpr, tdr, roc_auc with micro average scores added.
    '''
    fpr["micro"], tpr["micro"], _ = roc_curve(
        binarized_targets.ravel(), probabilities.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc


# adapted from scikit-learn examples
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def _roc_macro_average(fpr, tpr, roc_auc, classes):
    '''
    fpr: dict of false-positive rates for each class.
    tdr: dict of true-detection rates for each class.
    roc_auc: dict of auc scores for each class.
    classes: list of classes.

    Returns fpr, tdr, roc_auc with micro average scores added.
    '''
    # Aggregate all false positive rates for computing average
    all_fpr = np.unique(np.concatenate([fpr[c] for c in classes]))

    # Then interpolate all ROC curves at this point
    mean_tpr = np.zeros_like(all_fpr)
    for c in classes:
        mean_tpr += interp(all_fpr, fpr[c], tpr[c])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


# inspired by scikit-learn examples for multi-class ROC plots
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
def _roc_plot(fpr, tpr, roc_auc, classes, colors):
    '''
    fpr: dict of false-positive rates for each class.
    tdr: dict of true-detection rates for each class.
    roc_auc: dict of auc scores for each class.
    classes: list of classes.
    colors: list of colors.
    '''
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), sharey=True)
    lw = 3

    # plot averages in each panel
    for i in [0, 1]:
        axes[i].plot(fpr['micro'], tpr['micro'], color='navy', linestyle=':',
                     lw=lw,
                     label='micro-average (AUC = %0.2f)' % roc_auc['micro'])
        axes[i].plot(fpr['macro'], tpr['macro'], color='lightblue',
                     linestyle=':', lw=lw,
                     label='macro-average (AUC = %0.2f)' % roc_auc['macro'])
        # plot 1:1 ratio line
        axes[i].plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--',
                     label='Chance')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate')

    # left panel: averages only
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Receiver Operating Characteristic Average Scores')
    axes[0].legend(loc="lower right")

    # right panel: averages and per-class ROCs
    axes[1].set_title('Per-Class Receiver Operating Characteristics')

    for c, color in zip(classes, colors):
        plt.plot(fpr[c], tpr[c], color=color, lw=lw,
                 label='{0} (AUC = {1:0.2f})'.format(c, roc_auc[c]))
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return fig
