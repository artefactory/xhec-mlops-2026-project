"""The HTTP contract; values use the same measurement units as the training CSV."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Measurement = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class AbaloneFeatures(BaseModel):
    """Eight raw measurements. Targets and unknown fields are rejected."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "sex": "M",
                "length": 0.455,
                "diameter": 0.365,
                "height": 0.095,
                "whole_weight": 0.514,
                "shucked_weight": 0.2245,
                "viscera_weight": 0.101,
                "shell_weight": 0.15,
            }
        },
    )

    sex: Literal["M", "F", "I"]
    length: Measurement
    diameter: Measurement
    height: Measurement
    whole_weight: Measurement
    shucked_weight: Measurement
    viscera_weight: Measurement
    shell_weight: Measurement


class Prediction(BaseModel):
    """An age estimate, not a ring count or a confidence interval."""

    age_years: Annotated[float, Field(allow_inf_nan=False)]
    model_sha256: str


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    model_sha256: str
