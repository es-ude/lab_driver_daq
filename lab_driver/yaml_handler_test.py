import unittest
from os.path import join, exists
from dataclasses import dataclass
from lab_driver.yaml_handler import YamlConfigHandler
from lab_driver import get_repo_name, get_path_to_project


@dataclass
class SettingsTest:
    path: str
    val: int
    freq: float
    data: list
    meta: dict

DefaultSettingsTest = SettingsTest(
    path='test',
    val=1,
    freq=10.0,
    data=[0, 1, 2],
    meta={1: 'company', 2: 'street', 3: 'city'}
)

# --- DATA FOR TESTING
path2yaml = join(get_path_to_project(), join('temp_test', 'config'))
filename = 'Config_Test'
data_wr = {
    'Name': 'John Doe',
    'Position': 'DevOps Engineer',
    'Location': 'England',
    'Age': '26',
    'Experience': {'GitHub': 'Software Engineer', 'Google': 'Technical Engineer', 'Linkedin': 'Data Analyst'},
    'Languages': {'Markup': ['HTML'], 'Programming': ['Python', 'JavaScript', 'Golang']}
}


class TestYamlHandler(unittest.TestCase):
    dummy0 = YamlConfigHandler(
        yaml_template=data_wr,
        path2yaml=path2yaml,
        yaml_name=filename + '0'
    )
    dummy1 = YamlConfigHandler(
        yaml_template=DefaultSettingsTest,
        path2yaml=path2yaml,
        yaml_name=filename + '1'
    )

    def test_repo_name(self):
        test_name = ['driver_meas_dev', 'lab_driver_daq', 'lab_driver']
        check = get_repo_name()
        result = True if check in test_name else False
        self.assertTrue(result)

    def test_project_path(self):
        ref = ['driver_meas_dev', 'lab_driver_daq', 'lab_driver']
        chck = get_path_to_project()
        result = ref[0] in chck or ref[1] in chck or ref[2] in chck
        self.assertTrue(result)

    def test_yaml_create(self):
        self.dummy0.write_dict_to_yaml(data_wr)
        path2chck = join(path2yaml, f"{filename}0.yaml")
        self.assertTrue(exists(path2chck))

    def test_yaml_class(self):
        class_out = self.dummy1.get_class(SettingsTest)
        self.assertTrue(DefaultSettingsTest == class_out)

    def test_yaml_read(self):
        data_rd = self.dummy0.get_dict()
        self.assertTrue(data_wr == data_rd)


if __name__ == '__main__':
    unittest.main()
