import unittest
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
    test_suite = test_loader.discover('lab_driver', pattern='*_test.py')

    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)

    rmtree('temp_test')
    rmtree('config')
