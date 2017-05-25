# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from qiime2.plugin import SemanticType
from ..plugin_setup import plugin
from ._format import CoordinatesDirectoryFormat


SampleData = SemanticType('SampleData', field_names='type')

Coordinates = SemanticType('Coordinates', variant_of=SampleData.field['type'])

plugin.register_semantic_types(Coordinates)

plugin.register_semantic_type_to_format(
    SampleData[Coordinates],
    artifact_format=CoordinatesDirectoryFormat
)
