from scarlet.reduction import SASCurve, stitch_many, stitch_pair
from scarlet.reduction import stiching, stitching


def test_package_reexports_stitching_api() -> None:
    assert SASCurve.__module__ == "scarlet.reduction.stitching"
    assert stitch_pair.__module__ == "scarlet.reduction.stitching"
    assert stitch_many.__module__ == "scarlet.reduction.stitching"


def test_module_aliases_remain_importable() -> None:
    assert stiching.stitch_pair is stitch_pair
    assert stitching.stitch_many is stitch_many
