from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ActionClass(_message.Message):
    __slots__ = ["classId", "objectId", "sourceId"]
    CLASSID_FIELD_NUMBER: _ClassVar[int]
    OBJECTID_FIELD_NUMBER: _ClassVar[int]
    SOURCEID_FIELD_NUMBER: _ClassVar[int]
    classId: int
    objectId: int
    sourceId: int
    def __init__(self, objectId: _Optional[int] = ..., sourceId: _Optional[int] = ..., classId: _Optional[int] = ...) -> None: ...

class ActionResult(_message.Message):
    __slots__ = ["classes"]
    CLASSES_FIELD_NUMBER: _ClassVar[int]
    classes: _containers.RepeatedCompositeFieldContainer[ActionClass]
    def __init__(self, classes: _Optional[_Iterable[_Union[ActionClass, _Mapping]]] = ...) -> None: ...

class BatchDetectionResults(_message.Message):
    __slots__ = ["detectionResult", "process_time_ms"]
    DETECTIONRESULT_FIELD_NUMBER: _ClassVar[int]
    PROCESS_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    detectionResult: _containers.RepeatedCompositeFieldContainer[DetectionResults]
    process_time_ms: float
    def __init__(self, detectionResult: _Optional[_Iterable[_Union[DetectionResults, _Mapping]]] = ..., process_time_ms: _Optional[float] = ...) -> None: ...

class Detection(_message.Message):
    __slots__ = ["objectId", "points"]
    OBJECTID_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    objectId: int
    points: _containers.RepeatedCompositeFieldContainer[Point]
    def __init__(self, objectId: _Optional[int] = ..., points: _Optional[_Iterable[_Union[Point, _Mapping]]] = ...) -> None: ...

class DetectionResults(_message.Message):
    __slots__ = ["detection", "frameNum", "sourceId"]
    DETECTION_FIELD_NUMBER: _ClassVar[int]
    FRAMENUM_FIELD_NUMBER: _ClassVar[int]
    SOURCEID_FIELD_NUMBER: _ClassVar[int]
    detection: _containers.RepeatedCompositeFieldContainer[Detection]
    frameNum: int
    sourceId: int
    def __init__(self, detection: _Optional[_Iterable[_Union[Detection, _Mapping]]] = ..., frameNum: _Optional[int] = ..., sourceId: _Optional[int] = ...) -> None: ...

class Point(_message.Message):
    __slots__ = ["x", "y"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...
