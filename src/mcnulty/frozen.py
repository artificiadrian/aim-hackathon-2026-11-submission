"""Base classes of every model: immutable, typed."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)  # parses JSON text into a typed JsonValue


class Frozen(BaseModel):
    """Our own data: immutable, no silent type coercion, no unknown fields."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, strict=True, extra="forbid")


class FromText(BaseModel):
    """Data parsed from text at a boundary (command line): coercion from strings wanted, unknown fields not."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
