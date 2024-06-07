import os
from pathlib import Path
from model_api.model.classification import read_images, classify, IMAGE_WIDTH, IMAGE_HEIGHT


fixtures_path = Path(os.path.realpath(__file__)).parent / "fixtures"


def test_read_images_have_the_proper_shape():
    """Tests that the bytes images are converted to a numpy array of with the proper shape."""
    image_1 = (fixtures_path / "street.jpg").read_bytes()
    image_2 = (fixtures_path / "buildings.jpg").read_bytes()
    images_array = read_images([image_1, image_2], IMAGE_WIDTH, IMAGE_HEIGHT)
    assert images_array.shape == (2, IMAGE_WIDTH, IMAGE_HEIGHT, 3)


def test_classify_method_return_a_list_of_categories():
    """Test that the classify method returns a list of categories."""
    image_1 = (fixtures_path / "street.jpg").read_bytes()
    image_2 = (fixtures_path / "buildings.jpg").read_bytes()
    assert classify([image_1, image_2]) == [5, 0]
