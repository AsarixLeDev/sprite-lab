[CmdletBinding()]
param(
    [switch]$RunTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$ArgumentList = @()
    )

    Write-Host ""
    Write-Host "== $Label ==" -ForegroundColor Cyan
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Assert-ExistingPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required $Label is missing: $Path"
    }
}

$scriptDirectory = Split-Path -Parent $PSCommandPath
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDirectory "..\..")).Path
$pythonCommand = (Get-Command python -ErrorAction Stop).Source
$gitCommand = (Get-Command git -ErrorAction Stop).Source

$requiredPaths = [ordered]@{
    "source tree" = Join-Path $repoRoot "src"
    "frozen unlabeled r2 pool" = Join-Path $repoRoot "datasets\sprite_lab_unlabeled_pool_v1_r2"
    "legacy Dataset-v5 preview" = Join-Path $repoRoot "datasets\sprite_lab_multisource_v5_preview"
    "policy-v2 preview" = Join-Path $repoRoot "datasets\sprite_lab_multisource_v5_policy_v2_core_plus_weighted_sampling_preview"
    "Dataset-v5 contract" = Join-Path $repoRoot "experiments\v5_view_contract_v1"
}
foreach ($entry in $requiredPaths.GetEnumerator()) {
    Assert-ExistingPath -Path $entry.Value -Label $entry.Key
}

$previousPythonPath = $env:PYTHONPATH
$previousBytecodeSetting = $env:PYTHONDONTWRITEBYTECODE
$previousCudaVisibility = $env:CUDA_VISIBLE_DEVICES
$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:CUDA_VISIBLE_DEVICES = ""

Push-Location -LiteralPath $repoRoot
try {
    Write-Host "Sprite Lab safe capability snapshot" -ForegroundColor Green
    Write-Host "Repository: $repoRoot"
    Write-Host "This script never launches a GUI, provider inference, generation, training, production freeze, or promotion."

    Invoke-Checked -Label "Current branch" -FilePath $gitCommand -ArgumentList @("branch", "--show-current")
    Invoke-Checked -Label "Current commit" -FilePath $gitCommand -ArgumentList @("log", "-1", "--oneline")
    Invoke-Checked -Label "Working-tree status" -FilePath $gitCommand -ArgumentList @("status", "--short")

    Invoke-Checked -Label "Sprite Lab CLI help" -FilePath $pythonCommand -ArgumentList @("-m", "spritelab", "--help")
    Invoke-Checked -Label "Unlabeled-pool CLI help" -FilePath $pythonCommand -ArgumentList @("-m", "spritelab.unlabeled_pool", "--help")
    Invoke-Checked -Label "Dataset-v5 CLI help" -FilePath $pythonCommand -ArgumentList @("-m", "spritelab.dataset_v5.cli", "--help")
    Invoke-Checked -Label "Campaign-plan help" -FilePath $pythonCommand -ArgumentList @("-m", "spritelab", "train", "campaign-plan", "--help")
    Invoke-Checked -Label "Campaign-validate help" -FilePath $pythonCommand -ArgumentList @("-m", "spritelab", "train", "campaign-validate", "--help")
    Invoke-Checked -Label "Campaign-status help" -FilePath $pythonCommand -ArgumentList @("-m", "spritelab", "train", "campaign-status", "--help")
    Invoke-Checked -Label "Campaign-run help (display only)" -FilePath $pythonCommand -ArgumentList @("-m", "spritelab", "train", "campaign-run", "--help")
    Invoke-Checked -Label "Promotion-decision help (display only)" -FilePath $pythonCommand -ArgumentList @("-m", "spritelab", "eval", "promotion-decision", "--help")

    Invoke-Checked -Label "Verify frozen unlabeled r2 pool" -FilePath $pythonCommand -ArgumentList @(
        "-m", "spritelab.unlabeled_pool", "verify",
        "--pool", "datasets/sprite_lab_unlabeled_pool_v1_r2"
    )
    Invoke-Checked -Label "Verify immutable legacy Dataset-v5 preview" -FilePath $pythonCommand -ArgumentList @(
        "-m", "spritelab.dataset_v5.cli", "verify",
        "--dataset", "datasets/sprite_lab_multisource_v5_preview"
    )
    Invoke-Checked -Label "Verify policy-v2 core-plus-weighted preview" -FilePath $pythonCommand -ArgumentList @(
        "-m", "spritelab.dataset_v5.cli", "verify-policy-preview",
        "--dataset", "datasets/sprite_lab_multisource_v5_policy_v2_core_plus_weighted_sampling_preview"
    )
    Invoke-Checked -Label "Validate Dataset-v5 named-view contract" -FilePath $pythonCommand -ArgumentList @(
        "-m", "spritelab.dataset_v5.cli", "validate-contract",
        "--contract-root", "experiments/v5_view_contract_v1"
    )

    Write-Host ""
    Write-Host "== Human-readable reports ==" -ForegroundColor Cyan
    $reports = @(
        "experiments/label_v4_calibration_wave1/two_pass_remediation_v2/remediation_report.md",
        "experiments/v5_named_view_builder_v1/implementation_report.md",
        "experiments/v5_readiness_audit_v1/audit_report.md",
        "experiments/training_headless_architecture_remediation_v1/remediation_report.md",
        "experiments/training_campaign_orchestration_v1/remediation_report.md",
        "experiments/memorization_promotion_remediation_v1/remediation_report.md",
        "experiments/repo_capability_snapshot_v1/next_milestones.md"
    )
    foreach ($report in $reports) {
        $state = if (Test-Path -LiteralPath $report) { "present" } else { "missing" }
        Write-Host "[$state] $report"
    }

    if ($RunTests) {
        $testRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("spritelab-capability-tests-" + [guid]::NewGuid().ToString("N"))
        [System.IO.Directory]::CreateDirectory($testRoot) | Out-Null
        Write-Host ""
        Write-Host "Optional focused tests use OS-temporary fixtures: $testRoot" -ForegroundColor Yellow

        Invoke-Checked -Label "Labeling-v4 focused tests" -FilePath $pythonCommand -ArgumentList @(
            "-m", "pytest",
            "tests/test_label_v4_two_pass_workflow.py",
            "tests/test_label_v4_audit_prefill_gui.py",
            "-q", "--basetemp", (Join-Path $testRoot "label-v4")
        )
        Invoke-Checked -Label "Dataset-v5 selected tests" -FilePath $pythonCommand -ArgumentList @(
            "-m", "pytest", "tests", "-q", "-k", "dataset_v5",
            "--basetemp", (Join-Path $testRoot "dataset-v5")
        )
        Invoke-Checked -Label "Training infrastructure focused tests" -FilePath $pythonCommand -ArgumentList @(
            "-m", "pytest",
            "tests/test_training_architecture_ablation.py",
            "tests/test_training_campaign.py",
            "-q", "--basetemp", (Join-Path $testRoot "training")
        )
        Invoke-Checked -Label "Memorization and promotion selected tests" -FilePath $pythonCommand -ArgumentList @(
            "-m", "pytest", "tests", "-q", "-k", "memorization or promotion",
            "--basetemp", (Join-Path $testRoot "memorization")
        )
    }
    else {
        Write-Host ""
        Write-Host "Tests were not run. Re-run with -RunTests for the four focused selections; the full suite is never automatic." -ForegroundColor Yellow
    }
}
finally {
    Pop-Location
    if ($null -eq $previousPythonPath) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue } else { $env:PYTHONPATH = $previousPythonPath }
    if ($null -eq $previousBytecodeSetting) { Remove-Item Env:PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue } else { $env:PYTHONDONTWRITEBYTECODE = $previousBytecodeSetting }
    if ($null -eq $previousCudaVisibility) { Remove-Item Env:CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue } else { $env:CUDA_VISIBLE_DEVICES = $previousCudaVisibility }
}
