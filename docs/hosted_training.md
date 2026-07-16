# Hosted training

Hosted training uses the provider-neutral `spritelab.remote_compute.ComputeBackend` protocol. Every backend implements `probe`, `estimate`, `prepare`, `upload`, `launch`, `poll`, `stream_events`, `pause`, `cancel`, `resume`, `download_artifacts`, `verify_artifacts`, and `cleanup`.

All mutating remote operations use a stable idempotency key and an identity binding. A reused key with a different campaign/run identity fails closed. Cloud launches additionally require an explicit confirmation supplied to the launch call.

## Generic SSH machine

The SSH backend targets a Unix machine with OpenSSH and Python. It supports:

- explicit connection and remote-Python checks;
- an absolute validated remote workspace;
- hash-addressed, staged, atomically finalized uploads;
- local argument-array execution with `shell=False`;
- remote command argv encoded as JSON/base64 and decoded by a fixed Python shim;
- per-job state, event, and log files;
- reconnectable polling and streaming;
- graceful `SIGINT` pause and `SIGTERM` cancellation by process group;
- remote checkpoint hash verification;
- partial-file download, local hash verification, and atomic finalization;
- resume only after download, hash, and remote-identity verification;
- cleanup limited to adapter staging data, preserving run outputs.

Example project configuration (no secret values):

```yaml
compute:
  training:
    type: ssh
    host: gpu.example.net
    user: trainer
    port: 22
    workspace: /workspace/sprite-lab
    identity_file: ~/.ssh/id_ed25519
    cloud: true
```

Passwords, tokens, secrets, and private-key content are rejected. Authentication is provided by the runtime OpenSSH environment and key file, not stored in project configuration.

## RunPod status

The RunPod adapter is a complete lifecycle-shaped, tested **unavailable scaffold**. It validates GPU selection, image name, container/volume disk sizes, cloud type, shutdown policy, and the name of the environment variable holding the credential. It reports credential presence without returning the value. It performs zero provider calls.

Current official RunPod documentation establishes:

- bearer-authenticated Pod creation at `POST https://rest.runpod.io/v1/pods`: <https://docs.runpod.io/api-reference/pods/POST/pods>
- Pod listing and documented desired states `RUNNING`, `EXITED`, and `TERMINATED`: <https://docs.runpod.io/api-reference/pods/GET/pods>
- Pod deletion at `DELETE /pods/{podId}`: <https://docs.runpod.io/api-reference/pods/DELETE/pods/podId>
- the distinction between proxied SSH and public-IP SSH/SCP: <https://docs.runpod.io/pods/configuration/use-ssh>
- stop/start behavior and the warning that some stopped storage can continue billing: <https://docs.runpod.io/pods/manage-pods>

Safe end-to-end launch is unavailable because this repository does not define a reviewed immutable training image/template, bind Pod and SSH readiness to campaign identity, bind Pod identity to artifact/checkpoint identity, integrate a current provider quote, or reconcile stop/delete after a connection loss. The adapter therefore never creates a Pod and never fakes successful hosted training.

No volatile price is hardcoded. A price or estimated cost should be displayed only after a future implementation obtains a current, attributable provider quote.

Example validated scaffold configuration:

```yaml
compute:
  training:
    type: runpod
    api_key_env: RUNPOD_API_KEY
    image_name: organization/reviewed-image@sha256:replace-with-reviewed-digest
    gpu_type_ids:
      - NVIDIA RTX A6000
    gpu_count: 1
    container_disk_gb: 50
    volume_gb: 50
    shutdown_policy: terminate_after_artifact_verification
    cloud_type: SECURE
```

The API key itself must be supplied at runtime through the named environment variable.

## Other providers

Plugins can supply hosted backends through an instance-scoped `HostedBackendRegistry` and `create_plugin(hosted_backends=[...])`. A hosted backend must implement the complete protocol and declare `is_cloud = True`. There is no global registry mutation.

## Remote failures and cost uncertainty

Connection loss, provider unavailability, process failure, quota errors, insufficient disk, interrupted transfers, disappeared resources, and stale identities never become success states. When polling cannot establish resource state, the dashboard reports `resource_state_uncertain`, whether the resource may still accrue cost, and explicit guidance to inspect and stop or terminate it in the provider console.
