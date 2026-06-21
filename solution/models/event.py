from dataclasses import dataclass, field
from enum import Enum


class EventType(str, Enum):
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"


@dataclass
class IngestEvent:
    event_type: EventType
    doc_id: str
    text: str = ""
    metadata: dict = field(default_factory=dict)
