# Pydantic models for the web service

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InputData(BaseModel):
    """Physical measurements of one abalone (same units as the training dataset)."""

    model_config = ConfigDict(
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
        }
    )

    sex: Literal["M", "F", "I"] = Field(description="M (male), F (female) or I (infant)")
    length: float = Field(gt=0, description="Longest shell measurement (mm / 200)")
    diameter: float = Field(gt=0, description="Perpendicular to length (mm / 200)")
    height: float = Field(gt=0, description="With meat in shell (mm / 200)")
    whole_weight: float = Field(gt=0, description="Whole abalone (grams / 200)")
    shucked_weight: float = Field(gt=0, description="Weight of meat (grams / 200)")
    viscera_weight: float = Field(gt=0, description="Gut weight after bleeding (grams / 200)")
    shell_weight: float = Field(gt=0, description="Shell weight after drying (grams / 200)")


class PredictionOut(BaseModel):
    """Predicted number of rings and the corresponding age."""

    rings: float = Field(description="Predicted number of shell rings")
    age: float = Field(description="Predicted age in years (rings + 1.5)")
