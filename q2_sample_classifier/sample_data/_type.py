# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------


from qiime2.plugin import SemanticType
from q2_types.sample_data import SampleData


SampleData = SemanticType('SampleData', field_names='type')

Coordinates = SemanticType('Coordinates', variant_of=SampleData.field['type'])
