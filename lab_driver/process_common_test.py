import unittest
from lab_driver import get_path_to_project
from lab_driver.process_common import ProcessCommon


class TestDataAnalysis(unittest.TestCase):
    path2data = get_path_to_project(new_folder='test_data')

    def test_get_data_overview(self):
        hndl = ProcessCommon()
        ovr = hndl.get_data_overview(
            path=self.path2data,
            acronym='dac'
        )
        self.assertTrue(len(ovr) > 0)
