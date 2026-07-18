"""Server-prepared infrastructure smokes and a Playground-only checkpoint catalog."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from spritelab.training.smoke_bundle import (
    FALSE_ELIGIBILITY,
    SMOKE_EVIDENCE_SCHEMA,
    SMOKE_SCOPE,
    SMOKE_STATUS,
    SMOKE_WALL_CLOCK_LIMIT_SECONDS,
    SmokeBundleError,
    VerifiedSmokeBundle,
    anchored_directory,
    anchored_path_is_absent,
    artifact_bundle_directory,
    canonical_json_bytes,
    file_sha256,
    finalize_identity,
    load_device_receipt,
    load_plan,
    load_playground_registration,
    portable_relative_parts,
    prepare_smoke_environment_binding,
    prepare_smoke_interpreter_binding,
    prepare_smoke_orchestration_code_identity,
    prepare_smoke_runtime_closure,
    publish_evidence,
    publish_plan,
    publish_playground_snapshot,
    publish_run_container,
    read_stable_single_link_bytes,
    smoke_id_for_campaign,
    smoke_training_argv,
    smoke_worker_argv,
    stable_hash,
    validate_identity,
    verify_complete_bundle,
    verify_execution_guards,
)

_OPAQUE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{7,159}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_ID = re.compile(r"^exploratory-[0-9a-f]{24}$")


@dataclass(frozen=True)
class SmokePublicationRequest:
    conditioned_job_id: str
    candidate_identity_sha256: str
    publication_identity_sha256: str
    activation_manifest_sha256: str
    campaign_config_sha256: str
    campaign_identity_sha256: str

    def validate(self) -> None:
        if not _OPAQUE_ID.fullmatch(self.conditioned_job_id):
            raise SmokeBundleError("conditioned_job_id", "The conditioned Dataset-v5 job ID is invalid.")
        for value in (
            self.candidate_identity_sha256,
            self.publication_identity_sha256,
            self.activation_manifest_sha256,
            self.campaign_config_sha256,
            self.campaign_identity_sha256,
        ):
            if not _SHA256.fullmatch(value):
                raise SmokeBundleError("publication_identity", "A conditioned publication identity is malformed.")


@dataclass(frozen=True)
class SmokePreparationRequest(SmokePublicationRequest):
    preparation_nonce: str
    explicit_action: bool = False


@dataclass(frozen=True)
class SmokeRegistrationRequest(SmokePublicationRequest):
    smoke_id: str
    plan_identity: str
    cpu_receipt_identity: str
    cuda_receipt_identity: str
    explicit_action: bool = False


@dataclass(frozen=True)
class ExploratoryCheckpointCandidate:
    checkpoint_id: str
    registration_id: str
    smoke_id: str
    weights: str
    checkpoint_step: int
    checkpoint_sha256: str
    path: Path
    registration_identity: str
    evidence_identity: str
    campaign_identity: str
    code_identity: str
    freeze_identity: str
    purpose: str = "exploratory"

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "friendly_run_name": "Exploratory 2-step CUDA smoke",
            "checkpoint_step": self.checkpoint_step,
            "weights": self.weights,
            "status": SMOKE_STATUS,
            "purpose": self.purpose,
            "scope": SMOKE_SCOPE,
            "eligible": True,
            "playground_eligible": True,
            "production_eligible": False,
            "evaluation_eligible": False,
            "training_resume_eligible": False,
            "promotion_eligible": False,
            "warning": "Exploratory infrastructure smoke only — never production Evaluation or promotion evidence.",
        }


@dataclass(frozen=True)
class ExploratoryCheckpointCatalog:
    eligible: tuple[ExploratoryCheckpointCandidate, ...] = ()
    unavailable_count: int = 0
    default_checkpoint_id: str | None = None

    def find(self, checkpoint_id: str | None, *, weights: str | None = None) -> ExploratoryCheckpointCandidate | None:
        requested = checkpoint_id or self.default_checkpoint_id
        if requested is None:
            return None
        direct = next((item for item in self.eligible if item.checkpoint_id == requested), None)
        if direct is None:
            return None
        requested_weights = str(weights or direct.weights).lower()
        if direct.weights == requested_weights:
            return direct
        return next(
            (
                item
                for item in self.eligible
                if item.registration_id == direct.registration_id and item.weights == requested_weights
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": "Exploratory smoke checkpoints — Playground only",
            "default_checkpoint_id": self.default_checkpoint_id,
            "eligible": [item.to_dict() for item in self.eligible],
            "unavailable_count": self.unavailable_count,
            "production_catalog_merged": False,
        }


@dataclass(frozen=True)
class _PublicationContext:
    request: SmokePublicationRequest
    config_sha256: str
    activation: Any
    job: Mapping[str, Any]
    publication: Mapping[str, Any]
    bindings: Mapping[str, Any]


class ExploratorySmokeWorkflow:
    """Two explicit web actions: prepare an immutable plan, then register evidence."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        job_loader: Callable[[str], Mapping[str, Any]] | None = None,
        job_inventory_loader: Callable[[], list[Mapping[str, Any]]] | None = None,
        activation_loader: Callable[..., Any] | None = None,
        manifest_builder: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self._job_loader = job_loader
        self._job_inventory_loader = job_inventory_loader
        self._activation_loader = activation_loader
        self._manifest_builder = manifest_builder

    def eligible_publications(self) -> dict[str, Any]:
        """Passively list pre-activation publications using opaque server identities."""

        values: list[dict[str, Any]] = []
        for job in self._load_job_inventory():
            try:
                request = self._publication_request_from_job(job)
            except (TypeError, ValueError, SmokeBundleError):
                continue
            candidate = job.get("candidate") if isinstance(job.get("candidate"), Mapping) else {}
            values.append(
                {
                    "conditioned_job_id": request.conditioned_job_id,
                    "label": f"Conditioned publication {request.conditioned_job_id}",
                    "selected_images": int(candidate.get("selected_images", 0) or 0),
                    "configuration_activated": False,
                    "smoke_eligible": True,
                }
            )
        values.sort(key=lambda item: item["conditioned_job_id"], reverse=True)
        return {
            "schema_version": "spritelab.playground.smoke-publication-options.v1",
            "eligible": values,
            "count": len(values),
            "paths_exposed": False,
        }

    def prepared_plans(self) -> dict[str, Any]:
        """Passively discover immutable plans so a restarted page can reconstruct status."""

        root = self.project_root / "artifacts" / "training" / "smokes"
        try:
            if anchored_path_is_absent(root, self.project_root):
                return {"eligible": [], "count": 0, "paths_exposed": False}
            rows: list[tuple[int, dict[str, Any]]] = []
            with anchored_directory(root, self.project_root) as anchor:
                for name in anchor.names():
                    metadata = anchor.lstat(name)
                    if not re.fullmatch(r"smoke-[0-9a-f]{20}", name) or not stat.S_ISDIR(metadata.st_mode):
                        continue
                    try:
                        plan = load_plan(self.project_root, name)
                    except (OSError, TypeError, ValueError, SmokeBundleError):
                        continue
                    projection = self._plan_projection(plan)
                    projection["conditioned_job_id"] = str(dict(plan["bindings"])["conditioned_job_id"])
                    rows.append((int(getattr(metadata, "st_mtime_ns", 0)), projection))
        except (OSError, TypeError, ValueError, SmokeBundleError):
            return {"eligible": [], "count": 0, "paths_exposed": False}
        rows.sort(key=lambda item: (item[0], item[1]["smoke_id"]), reverse=True)
        values = [row for _mtime, row in rows]
        return {"eligible": values, "count": len(values), "paths_exposed": False}

    def prepare_job(self, job_id: str, preparation_nonce: str, *, explicit_action: bool) -> dict[str, Any]:
        request = self._publication_request_from_job(self._load_job(job_id))
        return self.prepare(
            SmokePreparationRequest(
                **request.__dict__,
                preparation_nonce=preparation_nonce,
                explicit_action=explicit_action,
            )
        )

    def validate_job_plan(self, job_id: str, smoke_id: str, plan_identity: str) -> dict[str, Any]:
        request = self._publication_request_from_job(self._load_job(job_id))
        context = self._publication_context(request, require_pre_activation=True)
        plan = load_plan(self.project_root, smoke_id)
        if plan.get("plan_identity") != plan_identity:
            raise SmokeBundleError("plan_identity_changed", "The selected smoke plan identity changed.")
        expected, _configurations, _manifests = self._build_plan(
            context,
            str(plan.get("preparation_nonce") or ""),
        )
        if plan != expected:
            raise SmokeBundleError("smoke_plan_stale", "Dataset, freeze, campaign, code, or config changed.")
        self._require_real_config_unchanged(plan)
        self._require_campaign_roots_unchanged(plan)
        return plan

    def register_job(
        self,
        job_id: str,
        smoke_id: str,
        plan_identity: str,
        *,
        explicit_action: bool,
        server_execution_identities: Mapping[str, str],
    ) -> dict[str, Any]:
        request = self._publication_request_from_job(self._load_job(job_id))
        plan = self.validate_job_plan(job_id, smoke_id, plan_identity)
        receipts = {device: load_device_receipt(self.project_root, plan, device) for device in ("cpu", "cuda")}
        return self.register(
            SmokeRegistrationRequest(
                **request.__dict__,
                smoke_id=smoke_id,
                plan_identity=plan_identity,
                cpu_receipt_identity=str(receipts["cpu"]["receipt_identity"]),
                cuda_receipt_identity=str(receipts["cuda"]["receipt_identity"]),
                explicit_action=explicit_action,
            ),
            server_execution_identities=server_execution_identities,
        )

    def prepare(self, request: SmokePreparationRequest) -> dict[str, Any]:
        request.validate()
        if request.explicit_action is not True:
            raise SmokeBundleError("explicit_smoke_prepare", "Smoke preparation requires an explicit action.")
        context = self._publication_context(request, require_pre_activation=True)
        plan, configurations, manifests = self._build_plan(context, request.preparation_nonce)
        verify_execution_guards(self.project_root, plan)
        try:
            publish_plan(
                self.project_root,
                plan,
                configurations=configurations,
                manifests=manifests,
            )
        except SmokeBundleError as exc:
            if exc.code != "publication_exists":
                raise
            existing = load_plan(self.project_root, str(plan["smoke_id"]))
            if existing != plan:
                raise SmokeBundleError(
                    "smoke_plan_conflict", "That smoke preparation belongs to different bytes."
                ) from exc
        try:
            publish_run_container(self.project_root, plan)
        except SmokeBundleError as exc:
            if exc.code != "publication_exists" or not self._run_container_matches(plan):
                raise SmokeBundleError("smoke_run_conflict", "The fixed smoke run container is unavailable.") from exc
        self._require_real_config_unchanged(plan)
        self._require_campaign_roots_unchanged(plan)
        return self._plan_projection(plan)

    def register(
        self,
        request: SmokeRegistrationRequest,
        *,
        server_execution_identities: Mapping[str, str],
    ) -> dict[str, Any]:
        request.validate()
        if request.explicit_action is not True:
            raise SmokeBundleError("explicit_smoke_registration", "Smoke registration requires an explicit action.")
        if not _SHA256.fullmatch(request.plan_identity):
            raise SmokeBundleError("plan_identity", "The smoke plan identity is malformed.")
        if not _SHA256.fullmatch(request.cpu_receipt_identity) or not _SHA256.fullmatch(request.cuda_receipt_identity):
            raise SmokeBundleError("receipt_identity", "The CPU/CUDA smoke receipt identity is malformed.")
        context = self._publication_context(request, require_pre_activation=True)
        plan = load_plan(self.project_root, request.smoke_id)
        if plan.get("plan_identity") != request.plan_identity:
            raise SmokeBundleError("plan_identity_changed", "The selected smoke plan identity changed.")
        expected_plan, _configs, _manifests = self._build_plan(
            context,
            str(plan.get("preparation_nonce") or ""),
        )
        if plan != expected_plan:
            raise SmokeBundleError("smoke_plan_stale", "Dataset, freeze, campaign, code, or config changed.")
        self._require_real_config_unchanged(plan)
        self._require_campaign_roots_unchanged(plan)
        bundle = verify_complete_bundle(self.project_root, plan)
        if (
            bundle.evidence["runs"]["cpu"].get("receipt_identity") != request.cpu_receipt_identity
            or bundle.evidence["runs"]["cuda"].get("receipt_identity") != request.cuda_receipt_identity
        ):
            raise SmokeBundleError("receipt_identity_changed", "A confirmed CPU/CUDA smoke receipt changed.")
        executions = dict(server_execution_identities)
        if set(executions) != {"cpu", "cuda"} or any(
            not _SHA256.fullmatch(str(value)) for value in executions.values()
        ):
            raise SmokeBundleError(
                "server_execution_identity",
                "The server-owned CPU/CUDA execution identities are incomplete.",
            )
        evidence_body = dict(bundle.evidence)
        evidence_body.pop("evidence_identity", None)
        evidence_body["server_execution_identities"] = executions
        bundle = VerifiedSmokeBundle(
            evidence=finalize_identity(evidence_body, "evidence_identity"),
            checkpoints=bundle.checkpoints,
        )
        self._require_real_config_unchanged(plan)
        self._require_campaign_roots_unchanged(plan)
        evidence_path = artifact_bundle_directory(self.project_root, request.smoke_id) / "smoke_evidence.json"
        try:
            publish_evidence(self.project_root, bundle.evidence)
        except FileExistsError:
            existing = self._read_json(evidence_path, max_bytes=64 * 1024 * 1024)
            if existing != dict(bundle.evidence):
                raise SmokeBundleError(
                    "smoke_evidence_conflict", "Existing smoke evidence has different content."
                ) from None
        _directory, registration = publish_playground_snapshot(self.project_root, bundle)
        return {
            "status": SMOKE_STATUS,
            "smoke_id": request.smoke_id,
            "plan_identity": request.plan_identity,
            "evidence_identity": bundle.evidence["evidence_identity"],
            "registration_id": registration["content_id"],
            "registration_identity": registration["registration_identity"],
            "purpose": "exploratory",
            "playground_eligible": True,
            **FALSE_ELIGIBILITY,
            "message": "Exploratory smoke checkpoint registered for Playground only; production Evaluation remains blocked.",
        }

    def catalog(self) -> ExploratoryCheckpointCatalog:
        """Passively reconstruct current registrations without importing Torch or mutating disk."""

        root = self.project_root / "runs" / "v3" / "playground" / "exploratory-checkpoints"
        try:
            if anchored_path_is_absent(root, self.project_root):
                return ExploratoryCheckpointCatalog()
        except (OSError, ValueError, SmokeBundleError):
            return ExploratoryCheckpointCatalog((), 1, None)
        candidates: list[ExploratoryCheckpointCandidate] = []
        unavailable = 0
        try:
            with anchored_directory(root, self.project_root) as anchor:
                names = anchor.names()
                for name in names:
                    try:
                        metadata = anchor.lstat(name)
                        if not _CONTENT_ID.fullmatch(name) or not stat.S_ISDIR(metadata.st_mode):
                            continue
                        candidates.extend(self._registration_candidates(name))
                    except (OSError, TypeError, ValueError, SmokeBundleError):
                        unavailable += 1
        except (OSError, ValueError):
            return ExploratoryCheckpointCatalog((), 1, None)
        candidates.sort(key=lambda item: (item.registration_id, item.weights == "ema"), reverse=True)
        default = next((item.checkpoint_id for item in candidates if item.weights == "ema"), None)
        return ExploratoryCheckpointCatalog(tuple(candidates), unavailable, default)

    def _registration_candidates(self, content_id: str) -> list[ExploratoryCheckpointCandidate]:
        registration = load_playground_registration(self.project_root, content_id)
        evidence_path = (
            artifact_bundle_directory(self.project_root, str(registration["smoke_id"])) / "smoke_evidence.json"
        )
        if file_sha256(evidence_path, boundary=self.project_root, max_bytes=64 * 1024 * 1024) != registration.get(
            "smoke_evidence_sha256"
        ):
            raise SmokeBundleError("smoke_evidence_changed", "Registered smoke evidence changed.")
        evidence = self._read_json(evidence_path, max_bytes=64 * 1024 * 1024)
        if evidence.get("schema_version") != SMOKE_EVIDENCE_SCHEMA:
            raise SmokeBundleError("smoke_evidence_schema", "Registered smoke evidence has the wrong schema.")
        validate_identity(evidence, "evidence_identity")
        if evidence.get("evidence_identity") != registration.get("evidence_identity"):
            raise SmokeBundleError("smoke_evidence_changed", "Registered smoke evidence identity changed.")
        executions = evidence.get("server_execution_identities")
        if (
            not isinstance(executions, Mapping)
            or set(executions) != {"cpu", "cuda"}
            or any(not _SHA256.fullmatch(str(value)) for value in executions.values())
        ):
            raise SmokeBundleError("server_execution_identity", "Server-owned smoke execution evidence is missing.")
        bindings = dict(registration.get("bindings") or {})
        publication_request = SmokePublicationRequest(
            conditioned_job_id=str(bindings.get("conditioned_job_id") or ""),
            candidate_identity_sha256=str(bindings.get("candidate_identity_sha256") or ""),
            publication_identity_sha256=str(bindings.get("publication_identity_sha256") or ""),
            activation_manifest_sha256=str(bindings.get("activation_manifest_sha256") or ""),
            campaign_config_sha256=str(bindings.get("campaign_config_sha256") or ""),
            campaign_identity_sha256=str(bindings.get("campaign_identity_sha256") or ""),
        )
        current = self._publication_context(publication_request, require_pre_activation=False)
        if dict(current.bindings) != bindings:
            raise SmokeBundleError("smoke_binding_stale", "Registered smoke bindings are no longer current.")
        directory = self.project_root / "runs" / "v3" / "playground" / "exploratory-checkpoints" / content_id
        values: list[ExploratoryCheckpointCandidate] = []
        for row in registration.get("checkpoints") or ():
            if not isinstance(row, Mapping):
                raise SmokeBundleError("registration_checkpoint", "Registered checkpoint inventory is invalid.")
            weights = str(row.get("weights") or "")
            expected_name = f"checkpoint_step_000002{'_ema' if weights == 'ema' else ''}.pt"
            if weights not in {"live", "ema"} or row.get("path") != expected_name or row.get("step") != 2:
                raise SmokeBundleError("registration_checkpoint", "Registered checkpoint inventory is invalid.")
            path = directory / expected_name
            checkpoint_bytes = read_stable_single_link_bytes(
                path,
                boundary=self.project_root,
                max_bytes=2 * 1024**3,
            )
            digest = hashlib.sha256(checkpoint_bytes).hexdigest()
            if digest != row.get("sha256") or len(checkpoint_bytes) != row.get("byte_count"):
                raise SmokeBundleError("registration_checkpoint_changed", "Registered checkpoint bytes changed.")
            checkpoint_id = f"{content_id}:{weights}"
            values.append(
                ExploratoryCheckpointCandidate(
                    checkpoint_id=checkpoint_id,
                    registration_id=content_id,
                    smoke_id=str(registration["smoke_id"]),
                    weights=weights,
                    checkpoint_step=2,
                    checkpoint_sha256=digest,
                    path=path,
                    registration_identity=str(registration["registration_identity"]),
                    evidence_identity=str(registration["evidence_identity"]),
                    campaign_identity=str(bindings["campaign_identity_sha256"]),
                    code_identity=str(bindings["training_code_identity_sha256"]),
                    freeze_identity=str(bindings["activation_manifest_sha256"]),
                )
            )
        if {item.weights for item in values} != {"live", "ema"}:
            raise SmokeBundleError("registration_checkpoint_pair", "Registered live/EMA pair is incomplete.")
        return values

    def _publication_context(
        self,
        request: SmokePublicationRequest,
        *,
        require_pre_activation: bool,
    ) -> _PublicationContext:
        request.validate()
        job = dict(self._load_job(request.conditioned_job_id))
        candidate = job.get("candidate")
        publication = job.get("publication")
        if (
            job.get("status") != "COMPLETE"
            or not isinstance(candidate, Mapping)
            or not isinstance(publication, Mapping)
        ):
            raise SmokeBundleError("conditioned_publication_incomplete", "The conditioned publication is incomplete.")
        activated = publication.get("configuration_activated") is True
        if require_pre_activation and publication.get("configuration_activated") is not False:
            raise SmokeBundleError(
                "conditioned_publication_already_activated",
                "Smoke evidence must be registered before production configuration activation.",
            )
        expected = {
            "candidate_identity": request.candidate_identity_sha256,
            "publication_identity_sha256": request.publication_identity_sha256,
            "activation_manifest_sha256": request.activation_manifest_sha256,
            "campaign_config_sha256": request.campaign_config_sha256,
            "campaign_identity_sha256": request.campaign_identity_sha256,
        }
        if candidate.get("candidate_identity") != expected.pop("candidate_identity") or any(
            publication.get(key) != value for key, value in expected.items()
        ):
            raise SmokeBundleError("conditioned_publication_changed", "The conditioned publication identity changed.")
        from spritelab.v3.config import ProjectConfig

        current = ProjectConfig.load(self.project_root, required=True)
        if current.path is None:
            raise SmokeBundleError("project_config_missing", "The project configuration is unavailable.")
        current_bytes = read_stable_single_link_bytes(
            current.path,
            boundary=self.project_root,
            max_bytes=16 * 1024 * 1024,
        )
        values = copy.deepcopy(current.values)
        activation_relative = _canonical_relative(str(publication.get("activation_manifest") or ""))
        campaign_relative = _canonical_relative(str(publication.get("campaign_config") or ""))
        if require_pre_activation and (
            values["dataset"].get("freeze_manifest") == activation_relative
            or values["training"].get("campaign_config") == campaign_relative
            or values["execution"].get("allow_training") is True
        ):
            raise SmokeBundleError(
                "production_config_already_activated",
                "The published campaign is no longer in the required pre-audit configuration state.",
            )
        if activated:
            if (
                values["dataset"].get("freeze_manifest") != activation_relative
                or values["training"].get("dataset_freeze") != activation_relative
                or values["training"].get("campaign_config") != campaign_relative
            ):
                raise SmokeBundleError(
                    "activated_publication_changed", "The activated conditioned publication no longer matches."
                )
            selected_config = current
            require_audit = True
        else:
            values["dataset"]["freeze_manifest"] = activation_relative
            values["training"]["dataset_freeze"] = activation_relative
            values["training"]["campaign_config"] = campaign_relative
            selected_config = ProjectConfig(root=current.root, path=current.path, values=values)
            require_audit = False
        activation = self._load_activation(selected_config, require_audit=require_audit)
        campaign = dict(activation.campaign)
        code_identity = campaign.get("code_identity")
        if (
            activation.freeze_sha256 != request.activation_manifest_sha256
            or activation.campaign_config_sha256 != request.campaign_config_sha256
            or campaign.get("campaign_identity") != request.campaign_identity_sha256
            or not isinstance(code_identity, Mapping)
            or not _SHA256.fullmatch(str(code_identity.get("sha256") or ""))
        ):
            raise SmokeBundleError("conditioned_activation_changed", "Freeze, campaign, or code identity changed.")
        artifacts = activation.artifacts
        identities = dict(campaign.get("identities") or {})
        evaluation = dict(campaign.get("evaluation") or {})
        artifact_hashes = {
            "dataset_view_manifest_sha256": file_sha256(Path(artifacts["view_manifest"]), boundary=self.project_root),
            "split_manifest_sha256": file_sha256(Path(artifacts["split_manifest"]), boundary=self.project_root),
            "conditioning_vocabulary_sha256": file_sha256(
                Path(artifacts["conditioning_vocabulary"]), boundary=self.project_root
            ),
            "benchmark_manifest_sha256": file_sha256(Path(artifacts["benchmark_manifest"]), boundary=self.project_root),
        }
        if (
            artifact_hashes["dataset_view_manifest_sha256"] != identities.get("dataset_view_manifest_hash")
            or artifact_hashes["split_manifest_sha256"] != identities.get("split_manifest_hash")
            or artifact_hashes["conditioning_vocabulary_sha256"] != identities.get("conditioning_vocabulary_hash")
            or artifact_hashes["benchmark_manifest_sha256"] != evaluation.get("benchmark_manifest_hash")
        ):
            raise SmokeBundleError("campaign_artifact_identity", "Campaign artifact identities are inconsistent.")
        bindings = {
            "conditioned_job_id": request.conditioned_job_id,
            "candidate_identity_sha256": request.candidate_identity_sha256,
            "publication_identity_sha256": request.publication_identity_sha256,
            "activation_manifest_sha256": request.activation_manifest_sha256,
            "campaign_config_sha256": request.campaign_config_sha256,
            "campaign_identity_sha256": request.campaign_identity_sha256,
            "training_code_identity_sha256": code_identity["sha256"],
            "training_code_identity": dict(code_identity),
            **artifact_hashes,
        }
        return _PublicationContext(
            request=request,
            config_sha256=hashlib.sha256(current_bytes).hexdigest(),
            activation=activation,
            job=job,
            publication=dict(publication),
            bindings=bindings,
        )

    def _build_plan(
        self,
        context: _PublicationContext,
        preparation_nonce: str,
    ) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
        campaign = dict(context.activation.campaign)
        runs = list(campaign.get("expected_runs") or ())
        if not runs or any(not isinstance(item, Mapping) for item in runs):
            raise SmokeBundleError("campaign_runs", "The bound campaign has no exact source run.")
        source = sorted((dict(item) for item in runs), key=lambda item: str(item.get("run_id")))[0]
        base = source.get("resolved_config")
        if not isinstance(base, Mapping) or source.get("resolved_config_sha256") != stable_hash(base):
            raise SmokeBundleError("source_config", "The campaign source configuration identity is invalid.")
        runtime = base.get("runtime")
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("determinism") != "strict"
            or int(runtime.get("max_steps", 0)) < 2
        ):
            raise SmokeBundleError("source_determinism", "The campaign source must require strict determinism.")
        smoke_id = smoke_id_for_campaign(context.request.campaign_identity_sha256, preparation_nonce)
        sentinels = self._campaign_root_sentinels(runs)
        configurations: dict[str, bytes] = {}
        manifests: dict[str, bytes] = {}
        configuration_records: dict[str, Any] = {}
        allowed_overrides: dict[str, Any] = {}
        software_version_override: Mapping[str, Any] | None = None
        hardware_summary_overrides: dict[str, Mapping[str, Any]] = {}
        if self._manifest_builder is None:
            from spritelab.training.experiment_system import hardware_summary, software_version

            software_version_override = software_version(self.project_root)
            hardware_summary_overrides = {device: hardware_summary(target_device=device) for device in ("cpu", "cuda")}
        for device in ("cpu", "cuda"):
            config = copy.deepcopy(dict(base))
            output_path = f"runs/v3/training-smokes/{smoke_id}/{device}"
            config["name"] = f"{base.get('name')}__exploratory_smoke_{device}"
            config_runtime = dict(config["runtime"])
            config_runtime["device"] = device
            config_runtime["out_dir"] = output_path
            config["runtime"] = config_runtime
            config_path = f"artifacts/training/smokes/{smoke_id}/configs/{device}.json"
            manifest_path = f"artifacts/training/smokes/{smoke_id}/configs/{device}.manifest.json"
            config_bytes = canonical_json_bytes(config, pretty=True)
            manifest = self._build_manifest(
                self.project_root / config_path,
                config,
                software_version_override=software_version_override,
                hardware_summary_override=hardware_summary_overrides.get(device),
            )
            manifest_bytes = canonical_json_bytes(manifest, pretty=True)
            configurations[device] = config_bytes
            manifests[device] = manifest_bytes
            public_environment, child_environment = prepare_smoke_environment_binding(
                self.project_root,
                smoke_id,
                device,
            )
            configuration_records[device] = {
                "config_path": config_path,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "manifest_path": manifest_path,
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "output_path": output_path,
                "environment": public_environment,
                "child_environment": child_environment,
                "wall_clock_limit_seconds": SMOKE_WALL_CLOCK_LIMIT_SECONDS[device],
                "writable_roots": [
                    f"artifacts/training/smokes/{smoke_id}/execution/{device}",
                    output_path,
                ],
            }
            allowed_overrides[device] = {
                "name": config["name"],
                "runtime.device": device,
                "runtime.out_dir": output_path,
            }
        plan = finalize_identity(
            {
                "schema_version": "spritelab.training.smoke-plan.v1",
                "smoke_id": smoke_id,
                "preparation_nonce": preparation_nonce,
                "status": "PREPARED",
                "scope": SMOKE_SCOPE,
                "purpose": "exploratory",
                "interpreter": prepare_smoke_interpreter_binding(),
                "orchestration_code": prepare_smoke_orchestration_code_identity(self.project_root),
                "runtime_closure": prepare_smoke_runtime_closure(self.project_root),
                "bindings": dict(context.bindings),
                "source": {
                    "source_run_id": source.get("run_id"),
                    "source_run_identity_sha256": source.get("run_identity"),
                    "source_resolved_config_sha256": source.get("resolved_config_sha256"),
                    "source_seed": source.get("seed"),
                },
                "derivation": {
                    "base_config_sha256": source.get("resolved_config_sha256"),
                    "allowed_overrides": allowed_overrides,
                    "smoke_cli_semantics": {
                        "steps": 2,
                        "batch_size_max": 2,
                        "sample_every": 0,
                        "save_every": 1,
                        "resume": False,
                        "unsafe_resume": False,
                    },
                },
                "config_sha256_before": context.config_sha256,
                "full_campaign_output_roots": sentinels,
                "configurations": configuration_records,
                "retry_policy": "A failed or interrupted device run is never resumed; prepare a fresh nonce/bundle.",
                **FALSE_ELIGIBILITY,
            },
            "plan_identity",
        )
        return plan, configurations, manifests

    def _build_manifest(
        self,
        path: Path,
        config: Mapping[str, Any],
        *,
        software_version_override: Mapping[str, Any] | None = None,
        hardware_summary_override: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        builder = self._manifest_builder
        builder_options: dict[str, Any] = {}
        if builder is None:
            from spritelab.training.cli.experiment_cmds import prepare_experiment_manifest

            builder = prepare_experiment_manifest
            if software_version_override is None or hardware_summary_override is None:
                raise SmokeBundleError(
                    "manifest_context",
                    "The server could not bind the smoke manifest's software and hardware context.",
                )
            builder_options = {
                "software_version_override": software_version_override,
                "hardware_summary_override": hardware_summary_override,
            }
        runtime = dict(config["runtime"])
        batch_size = min(2, int(runtime["batch_size"]))
        overrides = {
            "max_steps": 2,
            "batch_size": batch_size,
            "micro_batch_size": batch_size,
            "global_batch_size": batch_size,
            "effective_batch_size": batch_size * int(runtime.get("gradient_accumulation_steps", 1)),
            "sample_every": 0,
            "save_every": 1,
        }
        return dict(
            builder(
                path,
                write=False,
                config=config,
                runtime_overrides=overrides,
                resolution_root=self.project_root,
                **builder_options,
            )
        )

    def _campaign_root_sentinels(self, runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        sentinels: list[dict[str, Any]] = []
        for run in sorted(runs, key=lambda item: str(item.get("run_id"))):
            relative, path = _project_owned_path(self.project_root, str(run.get("output_root") or ""))
            if not anchored_path_is_absent(path, self.project_root):
                raise SmokeBundleError(
                    "production_output_not_fresh", "A full-campaign output root is no longer fresh and absent."
                )
            sentinels.append({"run_id": run.get("run_id"), "relative_path": relative, "state": "ABSENT"})
        return sentinels

    def _require_campaign_roots_unchanged(self, plan: Mapping[str, Any]) -> None:
        for sentinel in plan.get("full_campaign_output_roots") or ():
            if not isinstance(sentinel, Mapping) or sentinel.get("state") != "ABSENT":
                raise SmokeBundleError("campaign_sentinel", "A campaign output-root sentinel is invalid.")
            _relative, path = _project_owned_path(self.project_root, str(sentinel.get("relative_path") or ""))
            if not anchored_path_is_absent(path, self.project_root):
                raise SmokeBundleError(
                    "campaign_output_changed", "A full-campaign output root changed during smoke validation."
                )

    def _require_real_config_unchanged(self, plan: Mapping[str, Any]) -> None:
        from spritelab.v3.config import ProjectConfig

        current = ProjectConfig.load(self.project_root, required=True)
        if current.path is None:
            raise SmokeBundleError("project_config_missing", "The project configuration is unavailable.")
        actual = file_sha256(current.path, boundary=self.project_root, max_bytes=16 * 1024 * 1024)
        if actual != plan.get("config_sha256_before"):
            raise SmokeBundleError(
                "project_config_changed", "The real project configuration changed after preparation."
            )

    def _run_container_matches(self, plan: Mapping[str, Any]) -> bool:
        path = self.project_root / "runs" / "v3" / "training-smokes" / str(plan["smoke_id"]) / "state.json"
        try:
            state = self._read_json(path, max_bytes=16 * 1024 * 1024)
        except (OSError, SmokeBundleError):
            return False
        return (
            state.get("schema_version") == "spritelab.training.smoke-run-state.v1"
            and state.get("smoke_id") == plan["smoke_id"]
            and state.get("plan_identity") == plan["plan_identity"]
            and state.get("status") == "PREPARED"
        )

    def _plan_projection(self, plan: Mapping[str, Any]) -> dict[str, Any]:
        commands = []
        for device in ("cpu", "cuda"):
            config = dict(plan["configurations"])[device]
            commands.append(
                {
                    "device": device,
                    "environment": dict(config["environment"]),
                    "argv": smoke_training_argv(plan, device),
                    "worker_argv": smoke_worker_argv(plan, device),
                }
            )
        return {
            "status": "PREPARED",
            "smoke_id": plan["smoke_id"],
            "plan_identity": plan["plan_identity"],
            "commands": commands,
            "retry_policy": plan["retry_policy"],
            "portable_relative_paths": True,
            "private_or_absolute_paths_exposed": False,
            "full_campaign_started": False,
            **FALSE_ELIGIBILITY,
        }

    def _load_job(self, job_id: str) -> Mapping[str, Any]:
        if self._job_loader is not None:
            return self._job_loader(job_id)
        from spritelab.product_features.conditioned_v5.service import ConditionedDatasetService

        return ConditionedDatasetService(self.project_root).job(job_id)

    def _load_job_inventory(self) -> list[Mapping[str, Any]]:
        if self._job_inventory_loader is not None:
            return list(self._job_inventory_loader())
        from spritelab.product_features.conditioned_v5.service import ConditionedDatasetService

        inventory = ConditionedDatasetService(self.project_root).inventory()
        jobs = inventory.get("jobs")
        return list(jobs) if isinstance(jobs, list) else []

    def _publication_request_from_job(self, raw_job: Mapping[str, Any]) -> SmokePublicationRequest:
        job = dict(raw_job)
        candidate = job.get("candidate")
        publication = job.get("publication")
        if (
            job.get("status") != "COMPLETE"
            or not isinstance(candidate, Mapping)
            or not isinstance(publication, Mapping)
            or publication.get("configuration_activated") is not False
        ):
            raise SmokeBundleError(
                "conditioned_publication_ineligible",
                "Select a completed, published conditioned job that has not been activated.",
            )
        request = SmokePublicationRequest(
            conditioned_job_id=str(job.get("job_id") or ""),
            candidate_identity_sha256=str(candidate.get("candidate_identity") or ""),
            publication_identity_sha256=str(publication.get("publication_identity_sha256") or ""),
            activation_manifest_sha256=str(publication.get("activation_manifest_sha256") or ""),
            campaign_config_sha256=str(publication.get("campaign_config_sha256") or ""),
            campaign_identity_sha256=str(publication.get("campaign_identity_sha256") or ""),
        )
        request.validate()
        return request

    def _load_activation(self, prospective: Any, *, require_audit: bool) -> Any:
        if self._activation_loader is not None:
            return self._activation_loader(prospective, require_audit=require_audit)
        from spritelab.product_features.training.activation import load_conditioned_training_activation

        return load_conditioned_training_activation(
            prospective,
            require_audit=require_audit,
            require_activation_commit=False,
        )

    def _read_json(self, path: Path, *, max_bytes: int) -> dict[str, Any]:
        payload = read_stable_single_link_bytes(path, boundary=self.project_root, max_bytes=max_bytes)
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SmokeBundleError("smoke_json_invalid", "A smoke metadata document is invalid.") from exc
        if not isinstance(value, dict):
            raise SmokeBundleError("smoke_json_invalid", "A smoke metadata document is invalid.")
        return value


def _canonical_relative(value: str) -> str:
    try:
        parts = portable_relative_parts(value)
    except SmokeBundleError as exc:
        raise SmokeBundleError("publication_reference", "A server-owned publication reference is invalid.") from exc
    canonical = PurePosixPath(*parts).as_posix()
    if canonical != value:
        raise SmokeBundleError("publication_reference", "A server-owned publication reference is invalid.")
    return canonical


def _project_owned_path(root: Path, value: str) -> tuple[str, Path]:
    path = Path(value)
    candidate = path if path.is_absolute() else root.joinpath(*PurePosixPath(_canonical_relative(value)).parts)
    try:
        relative_path = candidate.relative_to(root)
        relative = PurePosixPath(*relative_path.parts).as_posix()
    except ValueError as exc:
        raise SmokeBundleError("campaign_output_root", "A campaign output root is outside this project.") from exc
    if not relative or any(part in {"", ".", ".."} for part in PurePosixPath(relative).parts):
        raise SmokeBundleError("campaign_output_root", "A campaign output root is unsafe.")
    return relative, root.joinpath(*PurePosixPath(relative).parts)


__all__ = [
    "ExploratoryCheckpointCandidate",
    "ExploratoryCheckpointCatalog",
    "ExploratorySmokeWorkflow",
    "SmokePreparationRequest",
    "SmokePublicationRequest",
    "SmokeRegistrationRequest",
]
