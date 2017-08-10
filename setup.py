# ----------------------------------------------------------------------------
# Copyright (c) 2017--, q2-sample-classifier development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from setuptools import find_packages, setup

import versioneer


setup(
    name='q2-sample-classifier',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='BSD-3-Clause',
    packages=find_packages(),
    author="Nicholas Bokulich",
    author_email="nbokulich@gmail.com",
    description="Machine learning classification and regression tools.",
    url="https://github.com/nbokulich/q2-sample-classifier",
    entry_points={
        'qiime2.plugins':
        ['q2-sample-classifier=q2_sample_classifier.plugin_setup:plugin']
    },
    package_data={
        'q2_sample_classifier.tests': ['data/*'],
        'q2_sample_classifier': ['assets/index.html'],
    },
    zip_safe=False,
)
