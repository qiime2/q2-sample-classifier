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

from .plugin_setup import plugin
from ._format import (SampleEstimatorDirFmt,
                      BooleanSeriesDirectoryFormat,
                      ImportanceDirectoryFormat,
                      PredictionsDirectoryFormat)

# Semantic Types
Predictions = SemanticType(
    'Predictions', variant_of=SampleData.field['type'])
SampleEstimator = SemanticType('SampleEstimator')
BooleanSeries = SemanticType(
    'BooleanSeries', variant_of=SampleData.field['type'])
Importance = SemanticType(
    'Importance', variant_of=FeatureData.field['type'])

# Registrations
plugin.register_semantic_types(
    SampleEstimator, BooleanSeries, Importance, Predictions)

plugin.register_semantic_type_to_format(
    SampleEstimator,
    artifact_format=SampleEstimatorDirFmt)

plugin.register_semantic_type_to_format(
    SampleData[BooleanSeries],
    artifact_format=BooleanSeriesDirectoryFormat)

plugin.register_semantic_type_to_format(
    SampleData[Predictions],
    artifact_format=PredictionsDirectoryFormat)

plugin.register_semantic_type_to_format(
    FeatureData[Importance],
    artifact_format=ImportanceDirectoryFormat)
