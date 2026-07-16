"""Fail-closed source provenance normalization and remediation."""

from spritelab.provenance.raw_remediation import (
    CONTROLLED_RESOLUTION_STATES,
    REQUIRED_SOURCE_BINDING_FIELDS,
    SourceResolutionError,
    compile_remediation,
    filter_candidate_records,
    render_download_recovery_script,
    verify_local_zip_candidate,
)

__all__ = [
    "CONTROLLED_RESOLUTION_STATES",
    "REQUIRED_SOURCE_BINDING_FIELDS",
    "SourceResolutionError",
    "compile_remediation",
    "filter_candidate_records",
    "render_download_recovery_script",
    "verify_local_zip_candidate",
]
