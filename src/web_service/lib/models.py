# Pydantic models for the web service: they validate the requests and the responses of the API.

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AbaloneInput(BaseModel):
    """Physical measurements of one abalone (lengths in mm / 200, weights in grams / 200)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "sex": "M",
                    "length": 0.455,
                    "diameter": 0.365,
                    "height": 0.095,
                    "whole_weight": 0.514,
                    "shucked_weight": 0.2245,
                    "viscera_weight": 0.101,
                    "shell_weight": 0.15,
                }
            ]
        }
    )

    sex: Literal["M", "F", "I"] = Field(
        description="M (male), F (female) or I (infant)"
    )
    length: float = Field(gt=0, description="Longest shell measurement")
    diameter: float = Field(gt=0, description="Perpendicular to the length")
    height: float = Field(gt=0, description="With meat in shell")
    whole_weight: float = Field(gt=0, description="Whole abalone")
    shucked_weight: float = Field(gt=0, description="Weight of the meat")
    viscera_weight: float = Field(gt=0, description="Gut weight after bleeding")
    shell_weight: float = Field(gt=0, description="After being dried")


class AbalonePrediction(BaseModel):
    """Predicted number of rings of the abalone (its age in years is about rings + 1.5)."""

    rings: int
