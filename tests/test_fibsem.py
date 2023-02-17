from fibsem_tools import read


def test_fibsem_v9_header_parse():
    test_dat_paths = ["tests/fixtures/v9_header_with_truncated_pixels.dat"]
    records = read(test_dat_paths)
    assert len(records) == 1

    header_dict = records[0].attrs
    header_keys = header_dict.keys()

    assert len(header_keys) == 99

    assert "SampleID" in header_keys
    assert "MillingYVoltage" in header_keys
