# ----------------------------------------------------------------------------
# Copyright (c) 2017-2018, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from qiime2.plugin import SemanticType
from q2_types.sample_data import SampleData
from q2_types.feature_data import FeatureData


Predictions = SemanticType(
    'Predictions', variant_of=SampleData.field['type'])
SampleEstimator = SemanticType('SampleEstimator')
BooleanSeries = SemanticType(
    'BooleanSeries', variant_of=SampleData.field['type'])
Importance = SemanticType(
    'Importance', variant_of=FeatureData.field['type'])
