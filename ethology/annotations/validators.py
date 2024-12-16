"""Validators for annotation files."""

import json
from pathlib import Path

import jsonschema
import jsonschema.exceptions
import jsonschema.validators
from attrs import define, field, validators


@define
class ValidJSON:
    """Class for validating JSON files.

    Attributes
    ----------
    path : pathlib.Path
        Path to the JSON file.

    schema : dict
        JSON schema to validate the file against.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON file cannot be decoded, or
        if the type of any of its keys does not match those
        specified in the schema.


    Notes
    -----
    https://json-schema.org/understanding-json-schema/

    """

    # Required attributes
    path: Path = field(validator=validators.instance_of(Path))

    # Optional attributes
    schema: dict | None = field(default=None)

    @path.validator
    def _file_is_json(self, attribute, value):
        """Ensure that the file is a JSON file."""
        try:
            with open(value) as file:
                json.load(file)
        except FileNotFoundError as not_found_error:
            raise FileNotFoundError(
                f"File not found: {value}."
            ) from not_found_error
        except json.JSONDecodeError as decode_error:
            raise ValueError(
                f"Error decoding JSON data from file: {value}."
            ) from decode_error

    @path.validator
    def _file_matches_JSON_schema(self, attribute, value):
        """Ensure that the JSON file matches the expected schema.

        The schema validation only checks the type for each specified
        key if it exists. It does not check for the presence of the keys.
        """
        # read json file
        with open(value) as file:
            data = json.load(file)

        # check against schema if provided
        if self.schema:
            try:
                jsonschema.validate(instance=data, schema=self.schema)
            except jsonschema.exceptions.ValidationError as val_err:
                # forward the error message as it is quite informative
                raise val_err


@define
class ValidVIAUntrackedJSON:
    """Class for validating VIA JSON files for untracked data.

    Checks the VIA JSON file for untracked data contains the required keys.

    Attributes
    ----------
    path : pathlib.Path
        Path to the JSON file.

    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def _file_contains_required_keys(self, attribute, value):
        """Ensure that the JSON file contains the required keys."""
        required_keys = {
            "main": ["_via_img_metadata", "_via_image_id_list"],
            "image_keys": ["filename", "regions"],
            "region_keys": ["shape_attributes", "region_attributes"],
            "shape_attributes_keys": ["x", "y", "width", "height"],
        }

        # Read data as dict
        with open(value) as file:
            data = json.load(file)

        # Check first level keys
        _check_keys(required_keys["main"], data)

        # Check keys in nested dicts
        for img_str, img_dict in data["_via_img_metadata"].items():
            # Check keys for each image dictionary
            _check_keys(
                required_keys["image_keys"],
                img_dict,
                additional_message=f" for {img_str}",
            )
            # Check keys for each region
            for i, region in enumerate(img_dict["regions"]):
                _check_keys(
                    required_keys["region_keys"],
                    region,
                    additional_message=f" for region {i} under {img_str}",
                )

                # Check keys under shape_attributes
                _check_keys(
                    required_keys["shape_attributes_keys"],
                    region["shape_attributes"],
                    additional_message=f" for region {i} under {img_str}",
                )


@define
class ValidCOCOUntrackedJSON:
    """Class for validating COCO JSON files for untracked data.

    The validator ensures that the file matches the expected schema.
    The schema validation only checks the type for each specified
    key if it exists. It does not check for the presence of the keys.

    Attributes
    ----------
    path : pathlib.Path
        Path to the JSON file.

    Raises
    ------
    ValueError
        If the JSON file does not match the expected schema.

    Notes
    -----
    https://json-schema.org/understanding-json-schema/

    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def _file_contains_required_keys(self, attribute, value):
        """Ensure that the JSON file contains the required keys."""
        required_keys = {
            "main": ["images", "annotations", "categories"],
            "image_keys": ["id", "file_name"],  # "height", "width"?
            "annotations_keys": ["id", "image_id", "bbox", "category_id"],
            "categories_keys": ["id", "name", "supercategory"],
        }

        # Read data as dict
        with open(value) as file:
            data = json.load(file)

        # Check first level keys
        _check_keys(required_keys["main"], data)

        # Check keys in images dicts
        for img_dict in data["images"]:
            _check_keys(
                required_keys["image_keys"],
                img_dict,
                additional_message=f" for image dict {img_dict}",
            )

        # Check keys in annotations dicts
        for annot_dict in data["annotations"]:
            _check_keys(
                required_keys["annotations_keys"],
                annot_dict,
                additional_message=f" for annotation dict {annot_dict}",
            )

        # Check keys in categories dicts
        for cat_dict in data["categories"]:
            _check_keys(
                required_keys["categories_keys"],
                cat_dict,
                additional_message=f" for category dict {cat_dict}",
            )


def _check_keys(
    list_required_keys: list[str],
    data_dict: dict,
    additional_message: str = "",
):
    missing_keys = set(list_required_keys) - data_dict.keys()
    if missing_keys:
        raise ValueError(
            f"Required key(s) {sorted(missing_keys)} not "
            f"found in {list(data_dict.keys())}" + additional_message + "."
        )
