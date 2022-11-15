from pdearena.data.twod.datapipes.common import ZarrLister


class TestZarrLister:
    def test_zarr_mix(self, mocker):
        mocker.patch("os.listdir", return_value=["a.zarr", "b.zarr", "c.zarr", "a", "b", "c"])
        dp = ZarrLister("/tmp")
        assert list(dp) == ["/tmp/a.zarr", "/tmp/b.zarr", "/tmp/c.zarr"]

    def test_nozarr(self, mocker):
        mocker.patch("os.listdir", return_value=["a", "b", "c"])
        dp = ZarrLister("/tmp")
        assert list(dp) == []
