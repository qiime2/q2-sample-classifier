---
title: 'q2-sample-classifier: machine-learning tools for microbiome classification and regression'
tags:
- microbiome
- supervised learning
- amplicon sequencing
- metagenomics
authors:
- name: Nicholas A Bokulich
  orcid: 0000-0002-1784-8935
  affiliation: 1
- name: Matthew R Dillon
  orcid: 0000-0002-7713-1952
  affiliation: 1
- name: Evan Bolyen
  orcid: 0000-0002-5362-6782
  affiliation: 1
- name: Benjamin D Kaehler
  orcid: 0000-0002-5318-9551
  affiliation: 2
- name: Gavin A Huttley
  orcid: 0000-0001-7224-2074
  affiliation: 2
- name: J Gregory Caporaso
  orcid: 0000-0002-8865-1670
  affiliation: "1, 3"
affiliations:
- name: The Pathogen and Microbiome Institute, Northern Arizona University, Flagstaff, AZ, USA
  index: 1
- name: Research School of Biology, Australian National University, Canberra, Australia
  index: 2
- name: Department of Biological Sciences, Northern Arizona University, Flagstaff, AZ, USA
  index: 3
date: 8 August 2018
bibliography: references.bib
---

# Summary
q2-sample-classifier is a plugin for the [QIIME 2](https://qiime2.org/) microbiome bioinformatics platform that facilitates access, reproducibility, and interpretation of supervised learning (SL) methods for a broad audience of non-bioinformatics specialists.

Microbiome studies often aim to predict outcomes or differentiate samples based on their microbial compositions, tasks that can be efficiently performed by SL methods [@Knights2011-ow]. The goal of SL is to train a machine learning model on a set of samples with known target values/class labels, and then use that model to predict the target values/class membership of additional, unlabeled samples. The ability to categorize new samples, as opposed to describing the structure of existing data, extends itself to many useful applications, e.g., the prediction of disease/susceptibility [@Yazdani2016-ih,Schubert2015-da,Pasolli2016-qi], crop productivity [@Chang2017-bq], wine chemical composition [@Bokulich2016-ea], or sample collection site [@Bokulich2013-go]; the identification of mislabeled samples in microbiome data sets [@Knights2011-ow]; or tracking microbiota-for-age development in children [@Subramanian2014-ch,Bokulich2016-wa].

We describe [q2-sample-classifier](https://github.com/qiime2/q2-sample-classifier), a [QIIME 2 plugin](https://qiime2.org/) to support SL tools for pattern recognition in microbiome data. This plugin provides several SL methods, automatic parameter tuning, feature selection, and various learning algorithms. The visualizations generated provide portable, shareable reports, publication-ready figures, and integrated decentralized data provenance. Additionally, integration as a QIIME 2 plugin streamlines data handling and supports the use of multiple user interfaces, including a prototype graphical user interface ([q2studio](https://github.com/qiime2/q2studio])), facilitating its use for non-expert users. The plugin is freely available under the BSD-3-Clause license at https://github.com/qiime2/q2-sample-classifier.

The q2-sample-classifier plugin is written in Python 3.5 and employs pandas [@McKinney2010-lu] and numpy [@Van_der_Walt2011-rv] for data manipulation, scikit-learn [@Pedregosa2011-vr] for SL and feature selection algorithms, scipy [@scipy] for statistical testing, and matplotlib [@Hunter2007-vy] and seaborn [@michael_waskom_2017_883859] for data visualization.

The standard workflow for classification and regression in q2-feature-classifier is shown in Figure 1. All q2-sample-classifier actions accept a feature table (i.e., matrix of feature counts per sample) and sample metadata (prediction targets) as input. Feature observations for q2-sample-classifier would commonly consist of microbial counts (e.g., amplicon sequence variants, operational taxonomic units, or taxa detected by marker-gene or shotgun metagenome sequencing methods), but any observation data, such as gene, transcript, protein, or metabolite abundance could be provided as input. Input samples are shuffled and split into training and test sets at a user-defined ratio (default: 4:1) with or without stratification (equal sampling per class label; stratified by default); test samples are left out of all model training steps and are only used for final model validation.

![Workflow schematic (A) and output data and visualizations (B-E) for q2-feature-classifier. Data splitting, model training, and testing (A) can be accompanied by automatic hyperparameter optimization (OPT) and recursive feature elimination for feature selection (RFE). Outputs include trained estimators for re-use on additional samples, lists of feature importance (B), RFE results if RFE is enabled (C), and predictions and accuracy results, including either confusion matrix heatmaps for classification results (D) or scatter plots of true vs. predicted values for regression results (E).](fig1.png)

The user can enable automatic feature selection and hyperparameter tuning, and can select the number of cross-validations to perform for each (default = 5). Feature selection is performed using cross-validated recursive feature elimination via scikit-learn’s [RFECV](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) to select the features that maximize predictive accuracy. Hyperparameter tuning is automatically performed using a cross-validated randomized parameter grid search via scikit-learn’s [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) to find hyperparameter permutations (within a sensible range) that maximize accuracy.

The following scikit-learn [@Pedregosa2011-vr] SL estimators are currently implemented in q2-sample-classifier: AdaBoost [@Freund1997-vv], Extra Trees [@Geurts2006-tv], Gradient boosting [@Friedman2002-vw], and Random Forest [@Breiman2001-ei] ensemble classifiers and regressors; linear SVC, linear SVR, and non-linear SVR support vector machine classifiers/regressors [@Cortes1995-jv]; k-Neighbors classifiers/regressors [@Altman1992-fo]; and Elastic Net [@Zou2005-py], Ridge [@Hoerl1970-sr], and Lasso [@Tibshirani1996-nt] regression models.

# Acknowledgments
The authors thank Jai Ram Rideout for his input and assistance integrating q2-sample-classifier with QIIME 2. This work was supported by the National Science Foundation [1565100 to JGC], and by the National Institutes of Health / National Cancer Institute Partnership for Native American Cancer Prevention [U54CA143924 and U54CA143925 to JGC].

# References
