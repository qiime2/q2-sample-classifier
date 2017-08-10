#!/usr/bin/env python

# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from sklearn.metrics import mean_squared_error, confusion_matrix

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import linregress, ttest_ind
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from itertools import combinations
from statsmodels.sandbox.stats.multicomp import multipletests


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


def _lmplot_from_dataframe(metadata, category, predicted_category, group_by,
                           plot_style="whitegrid"):
    sns.set_style(plot_style)
    g = sns.lmplot(category, predicted_category, data=metadata,
                   hue=group_by, fit_reg=False,
                   scatter_kws={"marker": ".", "s": 100}, legend=False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return g


def _boxplot_from_dataframe(metadata, category, dep, group_by,
                            plot_style="whitegrid"):
    sns.set_style(plot_style)
    ax = sns.boxplot(x=category, y=dep, hue=group_by, data=metadata)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylim(metadata[dep].min(), metadata[dep].max())
    return ax


def _clustermap_from_dataframe(table, metadata, group_by, category,
                               metric='correlation', method='weighted',
                               plot_style="whitegrid"):
    sns.set_style(plot_style)
    table = metadata[[group_by, category]].merge(
        table, left_index=True, right_index=True)
    table = table.groupby(by=[group_by, category]).median()

    # remove any empty columns
    table = table.loc[:, (table != 0).any(axis=0)]

    # generate cluster map
    g = sns.clustermap(table, metric=metric, method=method, z_score=1,
                       row_cluster=False)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), rotation=90)
    return g


def _filter_metadata_to_table_ids(table, metadata, dep, time, group_by):
    table = metadata[[dep, time, group_by]].merge(
        table, left_index=True, right_index=True)
    table = table[[dep, time, group_by]].dropna()
    return table


def _two_way_anova(table, metadata, dep, time, group_by):
    '''pd.DataFrame -> pd.DataFrame of AOV and OLS summary'''
    # Prep data
    table = _filter_metadata_to_table_ids(table, metadata, dep, time, group_by)

    # remove whitespace from column names
    table = table.rename(columns=lambda x: x.replace(' ', '_'))
    dep = dep.replace(' ', '_')
    time = time.replace(' ', '_')
    group_by = group_by.replace(' ', '_')

    # AOV
    mod = ols(formula='{0} ~ {1} * {2}'.format(dep, time, group_by),
              data=table).fit()
    aov_table = anova_lm(mod, typ=2)
    return aov_table, mod.summary2()


def _pairwise_stats(table, metadata, dep, time, group_by):
    '''pd.DataFrame -> pd.DataFrame
    Perform pairwise t-tests on all groups in group_by and time categories.
    '''
    # Prep data
    table = _filter_metadata_to_table_ids(table, metadata, dep, time, group_by)

    # find and store all valid subgroups' distributions of dependent var dep
    distributions = []
    for tp in table[time].unique():
        tab_tp = table[table[time] == tp]
        for group in tab_tp[group_by].unique():
            tab_group = tab_tp[tab_tp[group_by] == group][dep]
            distributions.append((tp, group, tab_group))

    # compare all distributions
    p_vals = []
    for combo in combinations(distributions, 2):
        try:
            t, p = ttest_ind(combo[0][2], combo[1][2], nan_policy='raise')
            p_vals.append(
                ((combo[0][0], combo[0][1]), (combo[1][0], combo[1][1]), t, p))
        except ValueError:
            pass

    result = pd.DataFrame(p_vals, columns=["Group A", "Group B", "t", "P"])
    result.set_index(['Group A', 'Group B'], inplace=True)
    result['q-value'] = multipletests(result['P'], method='fdr_bh')[1]
    result.sort_index(inplace=True)

    return result


def _linear_regress(actual, pred):
    '''Calculate linear regression on predicted versus expected values.
    actual: pandas.DataFrame
        Actual y-values for test samples.
    pred: pandas.DataFrame
        Predicted y-values for test samples.
    '''
    slope, intercept, r_value, p_value, std_err = linregress(actual, pred)
    mse = mean_squared_error(actual, pred)
    return pd.DataFrame([(mse, r_value, p_value, std_err, slope, intercept)],
                        columns=["Mean squared error", "R", "P-value",
                                 "Std Error", "Slope", "Intercept"],
                        index=[actual.name])


def _plot_confusion_matrix(y_test, y_pred, classes, accuracy, normalize=True,
                           palette='Default'):
    if palette == 'Default':
        palette = sns.cubehelix_palette(dark=0.15, light=0.95, as_cmap=True)

    cm = confusion_matrix(y_test, y_pred)
    # normalize
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    confusion = sns.heatmap(cm, cmap=palette)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    confusion.set_xticklabels(classes, rotation=90, ha='center')
    confusion.set_yticklabels(
        sorted(classes, reverse=True), rotation=0, ha='right')

    # generate confusion matrix as pd.DataFrame for viewing
    predictions = pd.DataFrame(cm, index=classes, columns=classes)
    # add empty row/column to show overall accuracy in bottom right cell
    # baseline error = error rate for a classifier that always guesses the
    # most common class
    n_samples = len(y_test)
    n_samples_largest_class = y_test.value_counts().iloc[0]
    basline_error = 1 - n_samples_largest_class / n_samples
    error_ratio = basline_error / (1 - accuracy)
    predictions["Overall Accuracy"] = ""
    predictions.loc["Overall Accuracy"] = ""
    predictions.loc["Baseline Error"] = ""
    predictions.loc["Accuracy Ratio"] = ""
    predictions.loc["Overall Accuracy"]["Overall Accuracy"] = accuracy
    predictions.loc["Baseline Error"]["Overall Accuracy"] = basline_error
    predictions.loc["Accuracy Ratio"]["Overall Accuracy"] = error_ratio

    return predictions, confusion


def _plot_RFE(rfecv):
    # If using fractional step, step = integer of fraction * n_features
    if rfecv.step < 1:
        rfecv.step = int(rfecv.step * len(rfecv.ranking_))
    # Need to manually calculate x-axis, as rfecv.grid_scores_ are a 1-d array
    x = [len(rfecv.ranking_) - (n * rfecv.step)
         for n in range(len(rfecv.grid_scores_)-1, -1, -1)]
    if x[0] < 1:
        x[0] = 1

    rfe = plt.figure()
    plt.xlabel("Feature Count")
    plt.ylabel("Accuracy")
    plt.plot(x, rfecv.grid_scores_, 'grey')
    return rfe
