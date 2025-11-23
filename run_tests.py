import unittest
from os.path import exists
from shutil import rmtree
from logging import basicConfig, DEBUG


def define_logger() -> None:
    basicConfig(
        level=DEBUG,
        filename='run_test_report.log',
        filemode='w',
        format='%(asctime)s - %(name)s - %(levelname)s = %(message)s'
    )


if __name__ == '__main__':
    define_logger()
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('elasticai/hw_measurements', pattern='*_test.py')

    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)
    if exists('temp_test'):
        rmtree('temp_test')
