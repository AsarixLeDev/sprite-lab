"""Instance-scoped registry for product/plugin-provided hosted backends."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from spritelab.remote_compute.contracts import (
    ComputeBackend,
    ComputeJob,
    ComputeJobRequest,
    PreparedCompute,
    ResumeRequest,
    verify_compute_job_request,
)

REQUIRED_BACKEND_METHODS = (
    "probe",
    "estimate",
    "prepare",
    "upload",
    "launch",
    "poll",
    "stream_events",
    "pause",
    "cancel",
    "resume",
    "download_artifacts",
    "verify_artifacts",
    "cleanup",
)


class ReceiptEnforcingComputeBackend:
    """Non-optional validation boundary around every plugin training backend."""

    def __init__(self, backend: ComputeBackend) -> None:
        self._backend = backend
        self.backend_id = backend.backend_id
        self.title = backend.title
        self.is_cloud = backend.is_cloud

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def prepare(self, context: Any, request: ComputeJobRequest) -> PreparedCompute:
        verify_compute_job_request(request, backend_id=self.backend_id)
        return self._backend.prepare(context, request)

    def launch(
        self, prepared: PreparedCompute, request: ComputeJobRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob:
        verify_compute_job_request(request, backend_id=self.backend_id)
        return self._backend.launch(prepared, request, cloud_confirmation=cloud_confirmation)

    def resume(
        self, prepared: PreparedCompute, resume: ResumeRequest, *, cloud_confirmation: bool = False
    ) -> ComputeJob:
        verify_compute_job_request(resume.request, backend_id=self.backend_id)
        return self._backend.resume(prepared, resume, cloud_confirmation=cloud_confirmation)


class HostedBackendRegistry:
    """No global provider registry; integrations pass an instance to the feature."""

    def __init__(self, backends: Iterable[ComputeBackend] = ()) -> None:
        self._backends: dict[str, ComputeBackend] = {}
        for backend in backends:
            self.register(backend)

    def register(self, backend: ComputeBackend) -> None:
        backend_id = str(getattr(backend, "backend_id", ""))
        if not backend_id:
            raise TypeError("Hosted backend must declare backend_id.")
        if backend_id in self._backends:
            raise ValueError(f"Duplicate hosted backend ID: {backend_id}")
        missing = [name for name in REQUIRED_BACKEND_METHODS if not callable(getattr(backend, name, None))]
        if missing:
            raise TypeError(f"Hosted backend {backend_id!r} is missing methods: {', '.join(missing)}")
        if not bool(getattr(backend, "is_cloud", False)):
            raise TypeError("Plugin-provided hosted backends must declare is_cloud=True.")
        self._backends[backend_id] = ReceiptEnforcingComputeBackend(backend)

    def get(self, backend_id: str) -> ComputeBackend | None:
        return self._backends.get(backend_id)

    def __iter__(self) -> Iterator[ComputeBackend]:
        return iter(self._backends.values())


def select_hosted_backend(registry: HostedBackendRegistry, backend_id: str) -> ComputeBackend:
    backend = registry.get(backend_id)
    if backend is None:
        available = ", ".join(item.backend_id for item in registry) or "none"
        raise LookupError(f"Hosted backend {backend_id!r} is not registered; available: {available}.")
    return backend
