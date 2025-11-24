from pathlib import Path
from unittest import TestCase, main
from .mxo4x import load_transient_data, load_fra_data

from elasticai.hw_measurements import get_path_to_project, TransientData, FrequencyResponse


class TestMXO4X(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path2start = Path(get_path_to_project("test_data")).absolute()

    def test_load_fra_data(self):
        path2file = self.path2start / "mxo4_fra.csv"
        assert path2file.exists() == True

        data = load_fra_data(
            path2file=path2file
        )
        assert type(data) == FrequencyResponse
        assert data.freq.shape == (67, )
        assert data.gain.shape == (67, )
        assert data.phase.shape == (67, )

    def test_load_transient_data(self):
        path2file = self.path2start / "mxo4_tran.h5"
        assert path2file.exists() == True

        data = load_transient_data(
            path2file=path2file,
            freq_ref=500.
        )
        assert type(data) == TransientData
        assert data.channels == ['C1', 'C2']
        assert data.num_channels == 2
        assert data.rawdata.shape == (2, 2000000)
        assert data.sampling_rate == 249999.75
        assert data.timestamps.shape == (2000000,)
        assert data.timestamps.size == data.rawdata.shape[1]
        assert data.num_channels == data.rawdata.shape[0]


if __name__ == "__main__":
    main()

