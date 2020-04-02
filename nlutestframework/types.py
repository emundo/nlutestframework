from typing import Optional, Dict, Union, List

DataSetTitle = str
FrameworkTitle = str
Intent = Optional[str]
ConfusionMatrix = Dict[Optional[str], Dict[Optional[str], int]]

# TODO: See: https://github.com/agronholm/sphinx-autodoc-typehints/issues/91
# Some types are ignored because the mypy checker complains on recursive types.
JSONSerializable = Union[          # type: ignore
    List["JSONSerializable"],      # type: ignore
    Dict[str, "JSONSerializable"], # type: ignore
    str,
    int,
    float,
    bool,
    None
]
