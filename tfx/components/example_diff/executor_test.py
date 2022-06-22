# Copyright 2022 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.components.example_diff.executor."""
import os
import tempfile

from absl.testing import absltest
import tensorflow_data_validation as tfdv
from tfx.components.example_diff import executor
from tfx.dsl.io import fileio
from tfx.proto import example_diff_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.types import standard_component_specs
from tfx.utils import json_utils

from google.protobuf import text_format

_EXECUTOR_TEST_PARAMS = [{
    'testcase_name': 'no_sharded_output',
    'sharded_output': False
}]
if tfdv.default_sharded_output_supported():
  _EXECUTOR_TEST_PARAMS.append({
      'testcase_name': 'yes_sharded_output',
      'sharded_output': True
  })


class ExecutorTest(absltest.TestCase):

  def get_temp_dir(self):
    return tempfile.mkdtemp()

  def _validate_skew_output(self, stats_path):
    # TODO(b/227361696): Validate contents.
    self.assertNotEmpty(fileio.glob(stats_path + '*-of-*'))

  def testDo(self):
    source_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'testdata')
    output_data_dir = os.path.join(
        os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
        self._testMethodName)
    fileio.makedirs(output_data_dir)

    config = text_format.Parse(
        """
      paired_example_skew: {
        identifier_features: 'trip_start_timestamp'
        identifier_features: 'company'
        skew_sample_size: 10
        allow_duplicate_identifiers: true
      }
    """, example_diff_pb2.ExampleDiffConfig())

    # Create input dict.
    examples = standard_artifacts.Examples()
    examples.uri = os.path.join(source_data_dir, 'csv_example_gen')
    examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval', 'test'])

    input_dict = {
        standard_component_specs.EXAMPLES_KEY: [examples],
        standard_component_specs.BASELINE_EXAMPLES_KEY: [examples],
    }

    exec_properties = {
        # List needs to be serialized before being passed into Do function.
        standard_component_specs.INCLUDE_SPLIT_PAIRS_KEY:
            json_utils.dumps([('train', 'train'), ('train', 'test')]),
        standard_component_specs.EXAMPLE_DIFF_CONFIG_KEY:
            config,
    }

    # Create output dict.
    example_diff = standard_artifacts.ExamplesDiff()
    example_diff.uri = output_data_dir
    output_dict = {
        standard_component_specs.EXAMPLE_DIFF_RESULT_KEY: [example_diff],
    }

    # Run executor.
    stats_gen_executor = executor.Executor()
    stats_gen_executor.Do(input_dict, output_dict, exec_properties)

    self.assertEqual(
        artifact_utils.encode_split_names(['train_test', 'train_train']),
        example_diff.split_names)
    self._validate_skew_output(
        os.path.join(example_diff.uri, 'Split-train_train', 'sample_pairs'))
    self._validate_skew_output(
        os.path.join(example_diff.uri, 'Split-train_train', 'skew_stats'))
    self._validate_skew_output(
        os.path.join(example_diff.uri, 'Split-train_test', 'sample_pairs'))
    self._validate_skew_output(
        os.path.join(example_diff.uri, 'Split-train_test', 'skew_stats'))


if __name__ == '__main__':
  absltest.main()
