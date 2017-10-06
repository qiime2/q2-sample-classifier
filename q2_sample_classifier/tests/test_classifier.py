# ----------------------------------------------------------------------------
# Copyright (c) 2017, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import qiime2
import pandas as pd
import numpy as np
from os import mkdir
from os.path import join
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning
from q2_sample_classifier.visuals import (
    _two_way_anova, _pairwise_stats, _linear_regress,
    _calculate_baseline_accuracy)
from q2_sample_classifier.classify import (
    classify_samples, regress_samples,
    maturity_index, detect_outliers, predict_coordinates)
from q2_sample_classifier.utilities import (
    split_optimize_classify, _set_parameters_and_estimator,
    _prepare_training_data, _optimize_feature_selection, _fit_and_predict,
    _calculate_feature_importances, _extract_important_features,
    _train_adaboost_base_estimator, _disable_feature_selection)
from q2_sample_classifier.plugin_setup import (
    CoordinatesFormat, CoordinatesDirectoryFormat, Coordinates,
    BooleanSeriesFormat, BooleanSeriesDirectoryFormat, BooleanSeries)
from q2_types.sample_data import SampleData
import tempfile
import shutil
import pkg_resources
from qiime2.plugin.testing import TestPluginBase
from qiime2.plugin import ValidationError
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC

filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=Warning)
filterwarnings("ignore", category=ConvergenceWarning)


class SampleClassifierTestPluginBase(TestPluginBase):
    package = 'q2_sample_classifier.tests'

    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix='q2-sample-classifier-test-temp-')

    def tearDown(self):
        self.temp_dir.cleanup()

    def get_data_path(self, filename):
        return pkg_resources.resource_filename(self.package,
                                               'data/%s' % filename)


class UtilitiesTests(SampleClassifierTestPluginBase):

    def setUp(self):
        super().setUp()

        self.exp_rf = pd.DataFrame(
            {'feature': ['a', 'b', 'c'], 'importance': [0.1, 0.2, 0.3]})

        self.exp_svm = pd.DataFrame(
            {'feature': ['a', 'b', 'c'], 'importance0': [0.1, 0.2, 0.3],
             'importance1': [0.4, 0.5, 0.6]})

        self.exp_lsvm = pd.DataFrame(
            {'feature': ['a', 'b', 'c'],
             'importance0': [-0.048794, -0.048794, -0.048794]})

        self.features = pd.DataFrame(
            {'a': [1, 1, 1, 1, 1], 'b': [1, 1, 1, 1, 1], 'c': [1, 1, 1, 1, 1]})

        self.targets = pd.Series(['a', 'a', 'b', 'b', 'a'], name='bullseye')

    def test_extract_important_features_1d_array(self):
        importances = _extract_important_features(
            self.features, np.ndarray((3,), buffer=np.array([0.1, 0.2, 0.3])))
        self.assertEqual(sorted(self.exp_rf), sorted(importances))

    def test_extract_important_features_2d_array(self):
        importances = _extract_important_features(
            self.features, np.ndarray(
                (2, 3), buffer=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])))
        self.assertEqual(sorted(self.exp_svm), sorted(importances))

    # test feature importance calculation with main classifier types
    def test_calculate_feature_importances_ensemble(self):
        estimator = RandomForestClassifier().fit(self.features, self.targets)
        fi = _calculate_feature_importances(self.features, estimator)
        self.assertEqual(sorted(self.exp_rf), sorted(fi))

    def test_calculate_feature_importances_svm(self):
        estimator = LinearSVC().fit(self.features, self.targets)
        fi = _calculate_feature_importances(self.features, estimator)
        self.assertEqual(sorted(self.exp_lsvm), sorted(fi))

    # confirm that feature selection incompatibility warnings work
    def test_disable_feature_selection_unsupported(self):
        with self.assertWarnsRegex(UserWarning, "does not support recursive"):
            _disable_feature_selection('KNeighborsClassifier', False)


class VisualsTests(SampleClassifierTestPluginBase):

    def test_two_way_anova(self):
        aov, mod_sum = _two_way_anova(tab1, md, 'Value', 'Time', 'Group')
        self.assertAlmostEqual(aov['PR(>F)']['Group'], 0.00013294988301061492)
        self.assertAlmostEqual(aov['PR(>F)']['Time'], 4.1672315658105502e-07)
        self.assertAlmostEqual(aov['PR(>F)']['Time:Group'], 0.0020603144625217)

    def test_pairwise_tests(self):
        res = _pairwise_stats(tab1, md, 'Value', 'Time', 'Group')
        self.assertAlmostEqual(
            res['q-value'][(1, 'a')][(1, 'b')], 0.066766544811987918)
        self.assertAlmostEqual(
            res['q-value'][(1, 'a')][(2, 'b')], 0.00039505928148818022)

    def test_linear_regress(self):
        res = _linear_regress(md['Value'], md['Time'])
        self.assertAlmostEqual(res.iloc[0]['Mean squared error'], 1.9413916666)
        self.assertAlmostEqual(res.iloc[0]['R'], 0.86414956372460128)
        self.assertAlmostEqual(res.iloc[0]['P-value'], 0.00028880275858705694)

    def test_calculate_baseline_accuracy(self):
        accuracy = 0.9
        y_test = pd.Series(['a', 'a', 'a', 'b', 'b', 'b'], name="class")
        classifier_accuracy = _calculate_baseline_accuracy(y_test, accuracy)
        expected_results = (6, 3, 0.5, 1.8)
        for i in zip(classifier_accuracy, expected_results):
            self.assertEqual(i[0], i[1])


class TestSemanticTypes(SampleClassifierTestPluginBase):

    def test_boolean_series_format_validate_positive(self):
        filepath = self.get_data_path('outliers.tsv')
        format = BooleanSeriesFormat(filepath, mode='r')
        format.validate()

    def test_boolean_series_format_validate_negative(self):
        filepath = self.get_data_path('coordinates.tsv')
        format = BooleanSeriesFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'BooleanSeriesFormat'):
            format.validate()

    def test_boolean_series_dir_fmt_validate_positive(self):
        filepath = self.get_data_path('outliers.tsv')
        shutil.copy(filepath, self.temp_dir.name)
        format = BooleanSeriesDirectoryFormat(self.temp_dir.name, mode='r')
        format.validate()

    def test_boolean_series_semantic_type_registration(self):
        self.assertRegisteredSemanticType(BooleanSeries)

    def test_sample_data_boolean_series_to_boolean_dir_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleData[BooleanSeries], BooleanSeriesDirectoryFormat)

    def test_pd_series_to_boolean_format(self):
        transformer = self.get_transformer(pd.Series, BooleanSeriesFormat)
        exp_index = pd.Index(['a', 'b', 'c', 'd'], dtype=object)
        exp = pd.Series([True, False, True, False],
                        name='outlier', index=exp_index)
        obs = transformer(exp)
        obs = pd.Series.from_csv(str(obs), sep='\t', header=0)
        self.assertEqual(sorted(exp), sorted(obs))

    def test_boolean_format_to_pd_series(self):
        _, obs = self.transform_format(
            BooleanSeriesFormat, pd.Series, 'outliers.tsv')
        exp_index = pd.Index(['a', 'b', 'c', 'd'], dtype=object)
        exp = pd.Series(['True', 'False', 'True', 'False'],
                        name='outlier', index=exp_index)
        self.assertEqual(sorted(exp), sorted(obs))

    def test_boolean_format_to_metadata(self):
        _, obs = self.transform_format(
            BooleanSeriesFormat, qiime2.Metadata, 'outliers.tsv')
        obs_category = obs.get_category('outlier')

        exp_index = pd.Index(['a', 'b', 'c', 'd'], dtype=object)
        exp = pd.Series(['True', 'False', 'True', 'False'],
                        name='outlier', index=exp_index)
        self.assertEqual(sorted(exp), sorted(obs_category.to_series()))

    def test_coordinates_format_validate_positive(self):
        filepath = self.get_data_path('coordinates.tsv')
        format = CoordinatesFormat(filepath, mode='r')
        format.validate()

    def test_coordinates_format_validate_negative(self):
        filepath = self.get_data_path('false-coordinates.tsv')
        format = CoordinatesFormat(filepath, mode='r')
        with self.assertRaisesRegex(ValidationError, 'CoordinatesFormat'):
            format.validate()

    def test_coordinates_dir_fmt_validate_positive(self):
        filepath = self.get_data_path('coordinates.tsv')
        shutil.copy(filepath, self.temp_dir.name)
        format = CoordinatesDirectoryFormat(self.temp_dir.name, mode='r')
        format.validate()

    def test_coordinates_semantic_type_registration(self):
        self.assertRegisteredSemanticType(Coordinates)

    def test_sample_data_coordinates_to_coordinates_dir_fmt_registration(self):
        self.assertSemanticTypeRegisteredToFormat(
            SampleData[Coordinates], CoordinatesDirectoryFormat)

    def test_pd_dataframe_to_coordinates_format(self):
        transformer = self.get_transformer(pd.DataFrame, CoordinatesFormat)
        exp = pd.DataFrame(
            {'Latitude': (38.306, 38.306), 'Longitude': (-122.228, -122.228)})
        obs = transformer(exp)
        obs = pd.DataFrame.from_csv(str(obs), sep='\t', header=0)
        self.assertEqual(sorted(exp), sorted(obs))

    def test_coordinates_format_to_pd_dataframe(self):
        _, obs = self.transform_format(
            CoordinatesFormat, pd.DataFrame, 'coordinates.tsv')
        exp = pd.DataFrame(
            {'Latitude': (38.306, 38.306, 38.306, 38.306),
             'Longitude': (-122.228, -122.228, -122.228, -122.228)},
            index=['a', 'b', 'c', 'd'])
        self.assertEqual(sorted(exp), sorted(obs))

    def test_coordinates_format_to_metadata(self):
        _, obs = self.transform_format(
            CoordinatesFormat, qiime2.Metadata, 'coordinates.tsv')
        obs_category = obs.get_category('Latitude')
        exp_index = pd.Index(['a', 'b', 'c', 'd'], dtype=object)
        exp = pd.Series(['38.306', '38.306', '38.306', '38.306'],
                        name='Latitude', index=exp_index)
        self.assertEqual(sorted(exp), sorted(obs_category.to_series()))


class EstimatorsTests(SampleClassifierTestPluginBase):

    def setUp(self):
        super().setUp()

        def _load_df(table_fp):
            table_fp = self.get_data_path(table_fp)
            table = qiime2.Artifact.load(table_fp)
            table = table.view(pd.DataFrame)
            return table

        def _load_md(md_fp):
            md_fp = self.get_data_path(md_fp)
            md = pd.DataFrame.from_csv(md_fp, sep='\t')
            md = qiime2.Metadata(md)
            return md

        def _load_mdc(md_fp, category):
            md_fp = self.get_data_path(md_fp)
            md = pd.DataFrame.from_csv(md_fp, sep='\t')
            md = qiime2.MetadataCategory(md[category])
            return md

        self.table_chard_fp = _load_df('chardonnay.table.qza')
        self.md_chard_fp = _load_md('chardonnay.map.txt')
        self.mdc_chard_fp = _load_mdc('chardonnay.map.txt', 'Region')
        self.table_ecam_fp = _load_df('ecam-table-maturity.qza')
        self.md_ecam_fp = _load_md('ecam_map_maturity.txt')
        self.mdc_ecam_fp = _load_mdc('ecam_map_maturity.txt', 'month')

    # test that the plugin/visualizer work
    def test_classify_samples(self):
        tmpd = join(self.temp_dir.name, 'RandomForestClassifier')
        mkdir(tmpd)
        classify_samples(tmpd, self.table_chard_fp, self.mdc_chard_fp,
                         test_size=0.5, cv=3,
                         n_estimators=2, n_jobs=1,
                         estimator='RandomForestClassifier',
                         parameter_tuning=True,
                         optimize_feature_selection=True)

    # test that each classifier works and delivers an expected accuracy result
    # when a random seed is set.
    def test_classifiers(self):
        for classifier in ['RandomForestClassifier', 'ExtraTreesClassifier',
                           'GradientBoostingClassifier', 'AdaBoostClassifier',
                           'LinearSVC', 'SVC', 'KNeighborsClassifier']:
            tmpd = join(self.temp_dir.name, classifier)
            mkdir(tmpd)
            estimator, pad, pt = _set_parameters_and_estimator(
                classifier, self.table_chard_fp, self.md_chard_fp, 'Region',
                n_estimators=10, n_jobs=1, cv=1,
                random_state=123, parameter_tuning=False, classification=True)
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_chard_fp, self.md_chard_fp, 'Region', estimator,
                tmpd, test_size=0.5, cv=1, random_state=123,
                n_jobs=1, optimize_feature_selection=False,
                parameter_tuning=False, param_dist=None,
                calc_feature_importance=False)
            self.assertAlmostEqual(accuracy, seeded_results[classifier])
            self.assertAlmostEqual(
                accuracy, seeded_results[classifier], places=4,
                msg='Accuracy of %s classifier was %f, but expected %f' % (
                    classifier, accuracy, seeded_results[classifier]))

    # test that the plugin/visualizer work
    def test_regress_samples(self):
        tmpd = join(self.temp_dir.name, 'RandomForestRegressor')
        mkdir(tmpd)
        regress_samples(tmpd, self.table_ecam_fp, self.mdc_ecam_fp,
                        test_size=0.5, cv=3,
                        n_estimators=2, n_jobs=1,
                        estimator='RandomForestRegressor')

    # test that each regressor works and delivers an expected accuracy result
    # when a random seed is set.
    def test_regressors(self):
        for regressor in ['RandomForestRegressor', 'ExtraTreesRegressor',
                          'GradientBoostingRegressor', 'AdaBoostRegressor',
                          'Lasso', 'Ridge', 'ElasticNet',
                          'KNeighborsRegressor', 'LinearSVR', 'SVR']:
            tmpd = join(self.temp_dir.name, regressor)
            mkdir(tmpd)
            estimator, pad, pt = _set_parameters_and_estimator(
                regressor, self.table_ecam_fp, self.md_ecam_fp, 'month',
                n_estimators=10, n_jobs=1, cv=1,
                random_state=123, parameter_tuning=False, classification=False)
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_ecam_fp, self.md_ecam_fp, 'month', estimator,
                tmpd, test_size=0.5, cv=1, random_state=123,
                n_jobs=1, optimize_feature_selection=False,
                parameter_tuning=False, param_dist=None, classification=False,
                calc_feature_importance=False, scoring=mean_squared_error)
            self.assertAlmostEqual(
                accuracy, seeded_results[regressor], places=4,
                msg='Accuracy of %s regressor was %f, but expected %f' % (
                    regressor, accuracy, seeded_results[regressor]))

    # test feature ordering
    # this bug emerged in maturity_index, where feature sorting was being
    # performed inadvertently during feature extraction (now fixed).
    # The issue was that scikit-learn handles dataframes of target data as
    # arrays without header information; hence, dataframes passed to an
    # estimator in different orders will cause misclassification. Here we
    # ensure that the labels in training and testing sets are passed in the
    # same order during split_optimize_classify (the following replicates a
    # minimal version of that function).
    def test_feature_ordering(self):
        # replicate minimal split_optimize_classify to extract importances
        estimator, pad, pt = _set_parameters_and_estimator(
            'RandomForestRegressor', self.table_ecam_fp, self.md_ecam_fp,
            'month', n_estimators=10, n_jobs=1, cv=1,
            random_state=123, parameter_tuning=False, classification=False)
        X_train, X_test, y_train, y_test = _prepare_training_data(
            self.table_ecam_fp, self.md_ecam_fp, 'month',
            test_size=0.1, random_state=123, load_data=True, stratify=False)
        X_train, X_test, importance = _optimize_feature_selection(
            self.temp_dir.name, X_train, X_test, y_train, estimator, cv=3,
            step=0.2, n_jobs=1)
        estimator, accuracy, y_pred = _fit_and_predict(
            X_train, X_test, y_train, y_test, estimator,
            scoring=mean_squared_error)
        # pull important features from a different dataframe
        importances = _calculate_feature_importances(X_train, estimator)
        table = self.table_ecam_fp.loc[:, importances["feature"]]
        # confirm ordering of feature (column) names
        ca = list(X_train.columns.values)
        cb = list(table.columns.values)
        self.assertEqual(ca, cb)

    # test adaboost base estimator trainer
    def test_train_adaboost_base_estimator(self):
        abe = _train_adaboost_base_estimator(
            self.table_chard_fp, self.mdc_chard_fp, 'Region',
            n_estimators=10, n_jobs=1, cv=3, random_state=None,
            parameter_tuning=True, classification=True)
        self.assertEqual(type(abe), AdaBoostClassifier)

    # test some invalid inputs/edge cases
    def test_invalids(self):
        estimator, pad, pt = _set_parameters_and_estimator(
            'RandomForestClassifier', self.table_chard_fp, self.md_chard_fp,
            'Region', n_estimators=10, n_jobs=1, cv=1,
            random_state=123, parameter_tuning=False, classification=True)
        regressor, pad, pt = _set_parameters_and_estimator(
            'RandomForestRegressor', self.table_chard_fp, self.md_chard_fp,
            'Region', n_estimators=10, n_jobs=1, cv=1,
            random_state=123, parameter_tuning=False, classification=True)
        # zero samples (if mapping file and table have no common samples)
        with self.assertRaisesRegex(ValueError, "metadata"):
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_ecam_fp, self.md_chard_fp, 'Region', estimator,
                self.temp_dir.name, test_size=0.5, cv=1, random_state=123,
                n_jobs=1, optimize_feature_selection=False,
                parameter_tuning=False, param_dist=None,
                calc_feature_importance=False)
        # too few samples to stratify
        with self.assertRaisesRegex(ValueError, "metadata"):
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_chard_fp, self.md_chard_fp, 'Region', estimator,
                self.temp_dir.name, test_size=0.9, cv=1, random_state=123,
                n_jobs=1, optimize_feature_selection=False,
                parameter_tuning=False, param_dist=None,
                calc_feature_importance=False)
        # regressor chosen for classification problem
        with self.assertRaisesRegex(ValueError, "convert"):
            estimator, cm, accuracy, importances = split_optimize_classify(
                self.table_chard_fp, self.md_chard_fp, 'Region', regressor,
                self.temp_dir.name, test_size=0.5, cv=1, random_state=123,
                n_jobs=1, optimize_feature_selection=False,
                parameter_tuning=False, param_dist=None,
                calc_feature_importance=False)

    # test experimental functions
    def test_maturity_index(self):
        maturity_index(self.temp_dir.name, self.table_ecam_fp, self.md_ecam_fp,
                       category='month', group_by='delivery', random_state=123,
                       n_jobs=1, control='Vaginal', test_size=0.4)

    def test_detect_outliers(self):
        detect_outliers(self.table_chard_fp, self.md_chard_fp,
                        random_state=123, n_jobs=1, contamination=0.05)

    def test_detect_outliers_with_subsets(self):
        detect_outliers(self.table_chard_fp, self.md_chard_fp,
                        random_state=123, n_jobs=1, contamination=0.05,
                        subset_category='Vineyard', subset_value=1)

    def test_detect_outliers_raise_error_on_missing_subset_data(self):
        with self.assertRaisesRegex(ValueError, "must both be provided"):
            detect_outliers(self.table_chard_fp, self.md_chard_fp,
                            random_state=123, n_jobs=1, contamination=0.05,
                            subset_category='Vineyard', subset_value=None)
        with self.assertRaisesRegex(ValueError, "must both be provided"):
            detect_outliers(self.table_chard_fp, self.md_chard_fp,
                            random_state=123, n_jobs=1, contamination=0.05,
                            subset_category=None, subset_value=1)

    def test_predict_coordinates(self):
        pred, coords = predict_coordinates(
            self.table_chard_fp, self.md_chard_fp,
            axis1_category='latitude', axis2_category='longitude',
            random_state=123, n_jobs=1)


md = pd.DataFrame([(1, 'a', 0.11), (1, 'a', 0.12), (1, 'a', 0.13),
                   (2, 'a', 0.19), (2, 'a', 0.18), (2, 'a', 0.21),
                   (1, 'b', 0.14), (1, 'b', 0.13), (1, 'b', 0.14),
                   (2, 'b', 0.26), (2, 'b', 0.27), (2, 'b', 0.29)],
                  columns=['Time', 'Group', 'Value'])

tab1 = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], columns=['Junk'])

seeded_results = {
    'RandomForestClassifier': 0.454545454545,
    'ExtraTreesClassifier': 0.454545454545,
    'GradientBoostingClassifier': 0.272727272727,
    'AdaBoostClassifier': 0.272727272727,
    'LinearSVC': 0.727272727273,
    'SVC': 0.545454545455,
    'KNeighborsClassifier': 0.363636363636,
    'RandomForestRegressor': 24.0533333333,
    'ExtraTreesRegressor': 16.1793650794,
    'GradientBoostingRegressor': 33.530579492,
    'AdaBoostRegressor': 27.746031746,
    'Lasso': 747.371448521,
    'Ridge': 521.402102726,
    'ElasticNet': 653.306453831,
    'KNeighborsRegressor': 44.7847619048,
    'LinearSVR': 511.816385601,
    'SVR': 72.6666666667}
