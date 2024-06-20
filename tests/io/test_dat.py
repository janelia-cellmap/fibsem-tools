from fibsem_tools import read


def test_fibsem_v9_header_parse():
    test_dat_path = "tests/fixtures/v9_header_with_truncated_pixels.dat"
    record = read(test_dat_path)

    header_dict = record.attrs
    header_keys = header_dict.keys()

    assert len(header_keys) == 99

    assert "SampleID" in header_keys
    assert "MillingYVoltage" in header_keys
