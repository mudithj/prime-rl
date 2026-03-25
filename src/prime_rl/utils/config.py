from pydantic import BaseModel
from pydantic_config import BaseConfig as BaseConfig  # noqa: F401
from pydantic_config import cli  # noqa: F401


<<<<<<< HEAD
def _convert_none(value):
    """Recursively convert None to ``"None"`` strings for TOML serialization."""
    if value is None:
        return "None"
    if isinstance(value, dict):
        return {k: _convert_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_none(item) for item in value]
    return value


def none_to_none_str(data: dict) -> dict:
    """Convert None values to ``"None"`` strings so they survive TOML serialization.

    TOML has no null type, so we use the ``"None"`` string convention which
    ``BaseConfig._none_str_to_none`` converts back to ``None`` on load.
    """
    return _convert_none(data)


=======
>>>>>>> parent of a25b3e7a1 (Enable none in config (#2093))
def get_all_fields(model: BaseModel | type) -> list[str]:
    if isinstance(model, BaseModel):
        model_cls = model.__class__
    else:
        model_cls = model

    fields = []
    for name, field in model_cls.model_fields.items():
        field_type = field.annotation
        fields.append(name)
        if field_type is not None and hasattr(field_type, "model_fields"):
            sub_fields = get_all_fields(field_type)
            fields.extend(f"{name}.{sub}" for sub in sub_fields)
    return fields
