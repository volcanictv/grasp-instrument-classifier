import numpy as np
import pytest

from grasp_classifier.preprocess import box_from_mask, crop_instance, pad_to_square


def test_pad_to_square_centers_content():
    crop = np.ones((10, 30, 3), dtype=np.uint8) * 255
    square = pad_to_square(crop)
    assert square.shape == (30, 30, 3)
    # centered: 10 rows of content start at row (30-10)//2 = 10
    assert (square[10:20, :, :] == 255).all()
    assert (square[0:10, :, :] == 0).all()
    assert (square[20:30, :, :] == 0).all()


def test_pad_to_square_already_square_is_unchanged():
    crop = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    square = pad_to_square(crop)
    assert square.shape == (20, 20, 3)
    assert (square == crop).all()


def test_crop_instance_without_mask_returns_raw_box():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[10:30, 10:30] = 255
    crop = crop_instance(image, box_xywh=(10, 10, 20, 20), mask=None, letterbox=False)
    arr = np.array(crop)
    assert arr.shape == (20, 20, 3)
    assert (arr == 255).all()


def test_crop_instance_with_mask_zeroes_outside_mask():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[15:25, 15:25] = True  # only the center of the box is "real" instrument
    crop = crop_instance(image, box_xywh=(10, 10, 20, 20), mask=mask, letterbox=False)
    arr = np.array(crop)
    assert arr.shape == (20, 20, 3)
    # pixels inside the mask (local coords 5:15, 5:15) stay 255, rest are zeroed
    assert (arr[5:15, 5:15] == 255).all()
    assert (arr[0:5, :] == 0).all()


def test_crop_instance_clips_box_at_frame_edge():
    image = np.ones((50, 50, 3), dtype=np.uint8) * 128
    # box extends past both edges of the frame
    crop = crop_instance(image, box_xywh=(40, 40, 30, 30), mask=None, letterbox=False)
    arr = np.array(crop)
    assert arr.shape == (10, 10, 3)  # clipped to the 10x10 region actually inside the frame


def test_crop_instance_letterbox_pads_elongated_crop_to_square():
    image = np.ones((100, 100, 3), dtype=np.uint8) * 200
    crop = crop_instance(image, box_xywh=(10, 10, 60, 10), mask=None, letterbox=True)
    arr = np.array(crop)
    assert arr.shape[0] == arr.shape[1] == 60


def test_crop_instance_no_letterbox_keeps_elongated_shape():
    image = np.ones((100, 100, 3), dtype=np.uint8) * 200
    crop = crop_instance(image, box_xywh=(10, 10, 60, 10), mask=None, letterbox=False)
    arr = np.array(crop)
    assert arr.shape[:2] == (10, 60)


def test_box_from_mask_matches_known_extent():
    mask = np.zeros((50, 50), dtype=bool)
    mask[5:15, 20:35] = True  # rows 5-14, cols 20-34
    x, y, w, h = box_from_mask(mask)
    assert (x, y, w, h) == (20, 5, 15, 10)


def test_box_from_mask_raises_on_empty_mask():
    mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError):
        box_from_mask(mask)
