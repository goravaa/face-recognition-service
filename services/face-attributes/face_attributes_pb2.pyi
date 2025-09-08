from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class AttributeRequest(_message.Message):
    __slots__ = ("image_data",)
    IMAGE_DATA_FIELD_NUMBER: _ClassVar[int]
    image_data: bytes
    def __init__(self, image_data: _Optional[bytes] = ...) -> None: ...

class AttributeResponse(_message.Message):
    __slots__ = ("status", "error_message", "race", "gender", "age", "race_probs", "gender_probs", "age_probs")
    class RaceProbsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class GenderProbsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    class AgeProbsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    RACE_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    AGE_FIELD_NUMBER: _ClassVar[int]
    RACE_PROBS_FIELD_NUMBER: _ClassVar[int]
    GENDER_PROBS_FIELD_NUMBER: _ClassVar[int]
    AGE_PROBS_FIELD_NUMBER: _ClassVar[int]
    status: str
    error_message: str
    race: str
    gender: str
    age: int
    race_probs: _containers.ScalarMap[str, float]
    gender_probs: _containers.ScalarMap[str, float]
    age_probs: _containers.ScalarMap[str, float]
    def __init__(self, status: _Optional[str] = ..., error_message: _Optional[str] = ..., race: _Optional[str] = ..., gender: _Optional[str] = ..., age: _Optional[int] = ..., race_probs: _Optional[_Mapping[str, float]] = ..., gender_probs: _Optional[_Mapping[str, float]] = ..., age_probs: _Optional[_Mapping[str, float]] = ...) -> None: ...
