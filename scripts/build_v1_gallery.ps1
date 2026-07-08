#Requires -Version 5.1
<#
.SYNOPSIS
    Quickstart wrapper for the official v1 demo/release gallery.

.DESCRIPTION
    Thin convenience wrapper around:
        python -m spritelab train build-v1-gallery
    using the official v1 checkpoint, CFG 3.0, 30 steps, and k16 deterministic
    palette projection. Never trains a model. See docs/v1_default.md.

.PARAMETER OutDir
    Output directory for samples, contact sheets, and the report.

.PARAMETER Device
    'cuda' or 'cpu'. Defaults to 'cuda' to match the validated release gallery.

.PARAMETER Seed
    Sampling seed. Defaults to the seed used for release validation.

.PARAMETER BatchSize
    Sampling batch size.

.PARAMETER PythonExe
    Path to the Python interpreter to use.

.EXAMPLE
    .\scripts\build_v1_gallery.ps1
    .\scripts\build_v1_gallery.ps1 -Device cpu -OutDir experiments\v1_gallery_cpu
#>
param(
    [string]$OutDir = "experiments\v1_gallery",
    [string]$Device = "cuda",
    [int]$Seed = 20260723,
    [int]$BatchSize = 32,
    [string]$PythonExe = "C:\Users\Mathieu\anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONPATH = "src"

& $PythonExe -m spritelab train build-v1-gallery `
    --out $OutDir `
    --device $Device `
    --seed $Seed `
    --batch-size $BatchSize
