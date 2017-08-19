# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import qiime2
import pandas as pd
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
    split_optimize_classify, _set_parameters_and_estimator)
import tempfile
import pkg_resources
from qiime2.plugin.testing import TestPluginBase
from sklearn.metrics import mean_squared_error


filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=Warning)
filterwarnings("ignore", category=ConvergenceWarning)


class SampleClassifierTestPluginBase(TestPluginBase):
    package = 'q2_sample_classifier.tests'

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix='q2-sample-classifier-test-temp-')

    def tearDown(self):
        self.temp_dir.cleanup()

    def get_data_path(self, filename):
        return pkg_resources.resource_filename(self.package,
                                               'data/%s' % filename)


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
            estimator, pd, pt = _set_parameters_and_estimator(
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
            estimator, pd, pt = _set_parameters_and_estimator(
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

    # test some invalid inputs/edge cases
    def test_invalids(self):
        estimator, pd, pt = _set_parameters_and_estimator(
            'RandomForestClassifier', self.table_chard_fp, self.md_chard_fp,
            'Region', n_estimators=10, n_jobs=1, cv=1,
            random_state=123, parameter_tuning=False, classification=True)
        regressor, pd, pt = _set_parameters_and_estimator(
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
                       category='month', group_by='delivery', n_jobs=1,
                       control='Vaginal', test_size=0.4)

    def test_detect_outliers(self):
        detect_outliers(self.table_chard_fp, self.md_chard_fp,
                        n_jobs=1, contamination=0.05)

    def test_predict_coordinates(self):
        pred, coords = predict_coordinates(
            self.table_chard_fp, self.md_chard_fp,
            axis1_category='latitude', axis2_category='longitude', n_jobs=1)


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
