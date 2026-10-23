from typing import Any, Dict, List
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


class RecordingHandler(BaseCallbackHandler):
    """Records chain starts (with tags/metadata) and custom events."""

    def __init__(self) -> None:
        self.starts: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []
        self.errors: List[BaseException] = []

    def on_chain_start(self, serialized, inputs, *, run_id: UUID, parent_run_id=None, tags=None, metadata=None, **kwargs):
        self.starts.append(
            {
                "name": kwargs.get("name"),
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags or [],
                "metadata": metadata or {},
            }
        )

    def on_chain_error(self, error, **kwargs):
        self.errors.append(error)

    def on_custom_event(self, name, data, *, run_id, tags=None, metadata=None, **kwargs):
        self.events.append({"name": name, "data": data, "run_id": run_id})

    def names(self) -> List[str]:
        return [s["name"] for s in self.starts]
