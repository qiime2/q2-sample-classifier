# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import SemanticType
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData


ClassifierPredictions = SemanticType(
    'ClassifierPredictions', variant_of=SampleData.field['type'])
RegressorPredictions = SemanticType(
    'RegressorPredictions', variant_of=SampleData.field['type'])
SampleEstimator = SemanticType('SampleEstimator', field_names='type')
Classifier = SemanticType(
    'Classifier', variant_of=SampleEstimator.field['type'])
Regressor = SemanticType(
    'Regressor', variant_of=SampleEstimator.field['type'])
BooleanSeries = SemanticType(
    'BooleanSeries', variant_of=SampleData.field['type'])
Importance = SemanticType(
    'Importance', variant_of=FeatureData.field['type'])
Probabilities = SemanticType(
    'Probabilities', variant_of=SampleData.field['type'])
TrueTargets = SemanticType(
    'TrueTargets', variant_of=SampleData.field['type'])
