"""HTTP client for the ComfyUI REST API.

Wraps the three endpoints zoomy needs — queue a workflow (``POST /prompt``),
read a prompt's history entry (``GET /history/<id>``), and interrupt the
running job (``POST /interrupt``) — plus a lightweight reachability probe.

The class also declares :class:`ConnectionProtocol`, the structural contract
the rendering and interface layers depend on, so tests can supply scripted
fakes without subclassing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import httpx

from zoomy.errors import ComfyConnectionError, ComfyRejectedWorkflowError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

CLIENT_IDENTIFIER = "zoomy"
REQUEST_TIMEOUT_SECONDS = 30.0
VALIDATION_FAILURE_STATUS_CODE = 400


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One prompt's history record from ComfyUI.

    Attributes:
        outputs: Node outputs keyed by node id; zoomy mostly ignores these
            because artifacts are discovered through the frame repository.
        status_messages: Status event pairs (``[name, payload]``) recorded by
            ComfyUI while the prompt ran, such as ``execution_error``.
        is_completed: Whether ComfyUI finished the prompt (successfully or
            not); execution failures are diagnosed from ``status_messages``.
    """

    outputs: dict[str, Any]
    status_messages: list[list[Any]]
    is_completed: bool


class ConnectionProtocol(Protocol):
    """Structural contract for anything that talks to ComfyUI for zoomy."""

    def is_reachable(self) -> bool:
        """Return True when ComfyUI answers a lightweight probe request."""
        ...

    def queue_workflow(self, workflow: Mapping[str, Any]) -> str:
        """Queue an API-format workflow and return its prompt identifier."""
        ...

    def fetch_history(self, prompt_identifier: str) -> HistoryEntry | None:
        """Return the prompt's history entry, or ``None`` when absent."""
        ...

    def interrupt(self) -> None:
        """Ask ComfyUI to interrupt the currently executing job."""
        ...


class ComfyConnection:
    """Small typed wrapper around the ComfyUI HTTP endpoints zoomy needs."""

    def __init__(self, address: str) -> None:
        """Store the ComfyUI base address (for example ``http://localhost:8188``)."""
        self.address = address.rstrip("/")
        self._client = httpx.Client(base_url=self.address, timeout=REQUEST_TIMEOUT_SECONDS)

    def is_reachable(self) -> bool:
        """Return True when ComfyUI answers a lightweight statistics request."""
        try:
            self._request("GET", "/system_stats")
        except ComfyConnectionError:
            return False
        return True

    def queue_workflow(self, workflow: Mapping[str, Any]) -> str:
        """Queue an API-format workflow and return its prompt identifier.

        Raises:
            ComfyRejectedWorkflowError: ComfyUI answered HTTP 400, meaning the
                workflow failed validation; the server's error and per-node
                details are attached.
            ComfyConnectionError: The request failed or the response was not the
                expected queue confirmation.
        """
        response = self._request(
            "POST", "/prompt", json={"prompt": workflow, "client_id": CLIENT_IDENTIFIER}
        )
        payload = response.json()
        prompt_identifier = payload.get("prompt_id")
        if not isinstance(prompt_identifier, str):
            raise ComfyConnectionError("ComfyUI queue response did not contain a prompt id")
        return prompt_identifier

    def fetch_history(self, prompt_identifier: str) -> HistoryEntry | None:
        """Return the prompt's history entry, or ``None`` when not present yet."""
        response = self._request("GET", f"/history/{prompt_identifier}")
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        record = payload.get(prompt_identifier)
        if not isinstance(record, dict):
            return None
        status = record.get("status")
        if not isinstance(status, dict):
            status = {}
        raw_messages = status.get("messages")
        if not isinstance(raw_messages, list):
            raw_messages = []
        messages = [list(message) for message in raw_messages if isinstance(message, list)]
        return HistoryEntry(
            outputs=record.get("outputs", {}),
            status_messages=messages,
            is_completed=bool(status.get("completed", False)),
        )

    def interrupt(self) -> None:
        """Ask ComfyUI to interrupt the currently executing job."""
        self._request("POST", "/interrupt")

    def _request(self, method: str, path: str, *, json: object | None = None) -> httpx.Response:
        """Perform one request, translating transport failures to zoomy errors.

        Raises:
            ComfyRejectedWorkflowError: On HTTP 400 with server detail.
            ComfyConnectionError: On any other transport or status failure.
        """
        try:
            response = self._client.request(method, path, json=json)
        except httpx.HTTPError as failure:
            raise ComfyConnectionError(
                f"Could not reach ComfyUI at {self.address}{path}: {failure}"
            ) from failure
        if response.status_code == VALIDATION_FAILURE_STATUS_CODE:
            raise _rejected_workflow_error(response)
        if not response.is_success:
            raise ComfyConnectionError(
                f"ComfyUI answered {response.status_code} for {path}: {response.text}"
            )
        return response


def _rejected_workflow_error(response: httpx.Response) -> ComfyRejectedWorkflowError:
    """Translate a 400 validation body into a rejection error."""
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    error = payload.get("error")
    if isinstance(error, dict):
        summary = str(error.get("message") or error.get("type") or "unknown validation failure")
    else:
        summary = "unknown validation failure"
    raw_node_errors = payload.get("node_errors")
    node_errors = raw_node_errors if isinstance(raw_node_errors, dict) else {}
    return ComfyRejectedWorkflowError(summary=summary, node_errors=node_errors)
