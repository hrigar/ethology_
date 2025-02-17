import json
from contextlib import nullcontext as does_not_raise

import pandas as pd
import pytest

from ethology.annotations.io.load_bboxes import from_files
from ethology.annotations.io.save_bboxes import (
    _validate_df_bboxes,
    to_COCO_file,
)


@pytest.mark.parametrize(
    "df, expected_exception, expected_error_message",
    [
        (
            [],
            pytest.raises(TypeError),
            "Expected a pandas DataFrame, but got <class 'list'>.",
        ),
        (
            pd.DataFrame(),
            pytest.raises(ValueError),
            "Expected index name to be 'annotation_id', but got 'None'.",
        ),
        (
            pd.DataFrame({"annotation_id": [1, 2, 3]}).set_index(
                "annotation_id"
            ),
            pytest.raises(ValueError),
            "Required bounding box coordinates "
            "'x_min', 'y_min', 'width', 'height', are not present in "
            "the dataframe.",
        ),
        (
            pd.DataFrame(
                {
                    "annotation_id": {0: 0, 1: 1, 2: 2},
                    "image_filename": {
                        0: "00000.jpg",
                        1: "00083.jpg",
                        2: "00166.jpg",
                    },
                    "image_id": {0: 0, 1: 83, 2: 166},
                    "x_min": {0: 963, 1: 376, 2: 458},
                    "y_min": {0: 283, 1: 314, 2: 329},
                    "width": {0: 302, 1: 301, 2: 301},
                    "height": {0: 172, 1: 123, 2: 131},
                    "supercategory": {0: "animal", 1: "animal", 2: "animal"},
                    "category": {0: "crab", 1: "crab", 2: "crab"},
                    "image_width": {0: 1280, 1: 1280, 2: 1280},
                    "image_height": {0: 720, 1: 720, 2: 720},
                }
            ).set_index("annotation_id"),  # data from "small_bboxes_COCO.json"
            does_not_raise(),
            "",
        ),
    ],
)
def test_validate_df_bboxes(
    df: pd.DataFrame,
    expected_exception: pytest.raises,
    expected_error_message: str,
):
    """Test _validate_df_bboxes throws the expected errors."""
    with expected_exception as excinfo:
        _validate_df_bboxes(df)
    if excinfo:
        assert expected_error_message == str(excinfo.value)


@pytest.mark.parametrize(
    "filename",
    [
        "small_bboxes_COCO.json",
        pytest.param(
            "COCO_JSON_sample_1.json",
            marks=pytest.mark.xfail(reason="should pass after PR48"),
        ),
    ],
)
def test_df_bboxes_to_COCO_file(filename, annotations_test_data, tmp_path):
    # Get input JSON file
    input_file = annotations_test_data[filename]

    # Read as bboxes dataframe
    df = from_files(input_file, format="COCO")

    # Export dataframe to COCO format
    output_file = to_COCO_file(df, output_filepath=tmp_path / "output.json")

    ########################################
    # Compare original and exported JSON files
    with open(input_file) as file:
        original_data = json.load(file)

    with open(output_file) as file:
        exported_data = json.load(file)

    ########################################
    # Check categories dictionaries match
    for categories_original, categories_exported in zip(
        original_data["categories"], exported_data["categories"], strict=True
    ):
        assert categories_original["id"] - 1 == categories_exported["id"]
        assert categories_original["name"] == categories_exported["name"]
        assert (
            categories_original["supercategory"]
            == categories_exported["supercategory"]
        )

    ########################################
    # Check images dictionaries match
    # Sort dicts in list of image dictionaries based on file_name
    original_img_dicts_sorted = sorted(
        original_data["images"], key=lambda x: x["file_name"]
    )
    exported_img_dicts_sorted = sorted(
        exported_data["images"], key=lambda x: x["file_name"]
    )

    # Compare the two lists of dictionaries
    for im_original, im_exported in zip(
        original_img_dicts_sorted,
        exported_img_dicts_sorted,
        strict=True,
    ):
        # Check common keys are equal
        common_keys = set(im_exported.keys()).intersection(im_original.keys())
        assert all(im_exported[ky] == im_original[ky] for ky in common_keys)

    ########################################
    # Check annotations
    # Sort annotations based on id
    exported_annot_dicts_sorted = sorted(
        exported_data["annotations"], key=lambda x: x["id"]
    )
    original_annot_dicts_sorted = sorted(
        original_data["annotations"], key=lambda x: x["id"]
    )

    # Compare the two lists of dictionaries
    for annot_original, annot_exported in zip(
        original_annot_dicts_sorted,
        exported_annot_dicts_sorted,
        strict=True,
    ):
        # Check common keys are equal
        common_keys = set(annot_exported.keys()).intersection(
            annot_original.keys()
        )
        assert all(
            annot_exported[ky] == annot_original[ky]
            for ky in common_keys
            if ky not in ["id", "category_id"]
            # "id" is expected to differ because we reindex
        )

        # Check category_id is as expected for COCO files exported
        # with VIA tool
        assert (
            annot_exported["category_id"] == annot_original["category_id"] - 1
        )
