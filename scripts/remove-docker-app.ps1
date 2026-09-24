$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$GeoParamsScriptDir = $PSScriptRoot
. (Join-Path $GeoParamsScriptDir "lib\common.ps1")

$ProjectLabel = "io.geoparams.project=geo-params-web"
$ManagerLabel = "io.geoparams.managed-by=geo-params-launcher"

function Get-LabeledIds {
    param([string[]]$Arguments)
    $ExitCode = Invoke-NativeCommand docker $Arguments -Capture
    if ($ExitCode -ne 0) {
        throw $NativeStdErr
    }
    return @($NativeStdOut -split "`r?`n" | Where-Object { $_ })
}

function Stop-ProjectContainers {
    $Ids = @(Get-LabeledIds @(
        "container", "ls", "-q",
        "--filter", "label=$ProjectLabel",
        "--filter", "label=$ManagerLabel"
    ))
    if ($Ids.Count -eq 0) {
        Write-Host "No running project containers were found."
        return
    }
    foreach ($Id in $Ids) {
        [void](Invoke-NativeCommand docker @("container", "stop", $Id))
    }
}

function Remove-ProjectContainers {
    $Ids = @(Get-LabeledIds @(
        "container", "ls", "-aq",
        "--filter", "label=$ProjectLabel",
        "--filter", "label=$ManagerLabel"
    ))
    if ($Ids.Count -eq 0) {
        Write-Host "No project-labeled containers were found."
        return
    }
    foreach ($Id in $Ids) {
        [void](Invoke-NativeCommand docker @("container", "rm", "-f", $Id))
    }
}

function Remove-ProjectNetworks {
    $Ids = @(Get-LabeledIds @(
        "network", "ls", "-q",
        "--filter", "label=$ProjectLabel",
        "--filter", "label=$ManagerLabel"
    ))
    if ($Ids.Count -eq 0) {
        Write-Host "No project-labeled networks were found."
        return
    }
    foreach ($Id in $Ids) {
        [void](Invoke-NativeCommand docker @("network", "rm", $Id))
    }
}

Ensure-Docker
Write-Host "GeoParams Web - Docker cleanup"
Write-Host "  1) Stop the application and keep its containers"
Write-Host "  2) Remove its containers and private network"
Write-Host "  3) Also remove images built and labeled by this project"
Write-Host "  q) Cancel"
$Choice = Read-MenuChoice "Choice" "q"

switch ($Choice) {
    "1" { Stop-ProjectContainers }
    "2" {
        Remove-ProjectContainers
        Remove-ProjectNetworks
    }
    "3" {
        Remove-ProjectContainers
        Remove-ProjectNetworks
        $Ids = @(Get-LabeledIds @(
            "image", "ls",
            "--filter", "label=$ProjectLabel",
            "--filter", "label=$ManagerLabel",
            "--format", "{{.ID}}"
        ) | Sort-Object -Unique)
        if ($Ids.Count -eq 0) {
            Write-Host "No project-labeled images were found."
        } else {
            foreach ($Id in $Ids) {
                [void](Invoke-NativeCommand docker @("image", "rm", $Id))
            }
        }
    }
    { $_ -match '^[qQ]$' } {
        Write-Host "Nothing was changed."
        exit 0
    }
    default {
        Write-Host "Invalid choice. Nothing was changed."
        exit 1
    }
}

Write-Host "Saved uploads, results, and downloaded datasets were not deleted."
