# ----------------------------------------------------------------------------
# Copyright (c) 2017-2022, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import tempfile
import pkg_resources

from qiime2.plugin.testing import TestPluginBase


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
