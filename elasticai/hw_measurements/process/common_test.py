import unittest
from elasticai.hw_measurements import get_path_to_project
from elasticai.hw_measurements.process.common import ProcessCommon


class TestDataAnalysis(unittest.TestCase):
    path2data = get_path_to_project(new_folder='test_data')

    def test_get_data_overview(self):
        hndl = ProcessCommon()
        ovr = hndl.get_data_overview(
            path=self.path2data,
            acronym='dac'
        )
        self.assertTrue(len(ovr) > 0)
