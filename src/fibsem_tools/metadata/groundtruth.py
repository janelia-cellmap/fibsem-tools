from pydantic import BaseModel
from typing import List


class InstanceName(BaseModel):
    long: str
    short: str


class AnnotationState(BaseModel):
    present: bool
    annotated: bool


class Label(BaseModel):
    value: int
    name: InstanceName
    annotationState: AnnotationState


class LabelList(BaseModel):
    labels: List[Label]
