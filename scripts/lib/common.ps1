$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

if (-not (Get-Variable GeoParamsScriptDir -ErrorAction SilentlyContinue)) {
    throw "Set GeoParamsScriptDir before loading common.ps1"
}

$RepoDir = Split-Path -Parent $GeoParamsScriptDir
$AppDir = Join-Path $RepoDir "geo_params_web"
$DockerUrl = "https://docs.docker.com/get-started/get-docker/"
$DatasetUrl = "https://drive.google.com/drive/folders/" +
    "1s-NAWbgukQG-1Q3M5MpO808XRqA1QVw4?usp=sharing"
$ComposeStyle = $null
$NativeStdOut = ""
$NativeStdErr = ""

if (-not $env:COMPOSE_PROJECT_NAME) {
    $env:COMPOSE_PROJECT_NAME = "geo-params-web"
}
if (-not $env:GEO_PARAMS_IMAGE) {
    $env:GEO_PARAMS_IMAGE = "geo-params-web-app:app-v2"
}
if (-not $env:GEO_PARAMS_DATASET_IMAGE) {
    $env:GEO_PARAMS_DATASET_IMAGE = "geo-params-datasets:app-v2"
}

function ConvertTo-NativeArgument {
    param([string]$Argument)
    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }
    $Escaped = $Argument -replace '(\\*)"', '$1$1\"'
    $Escaped = $Escaped -replace '(\\+)$', '$1$1'
    return '"' + $Escaped + '"'
}

function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FileName,
        [string[]]$Arguments = @(),
        [switch]$Capture,
        [string]$LogFile
    )
    $Info = New-Object System.Diagnostics.ProcessStartInfo
    $Info.FileName = $FileName
    $Info.Arguments = (($Arguments | ForEach-Object {
        ConvertTo-NativeArgument ([string]$_)
    }) -join ' ')
    $Info.UseShellExecute = $false
    $Info.WorkingDirectory = (Get-Location).ProviderPath
    $Info.CreateNoWindow = $false
    $Redirect = $Capture -or [bool]$LogFile
    $Info.RedirectStandardOutput = $Redirect
    $Info.RedirectStandardError = $Redirect

    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $Info
    [void]$Process.Start()
    if ($Redirect) {
        $OutTask = $Process.StandardOutput.ReadToEndAsync()
        $ErrTask = $Process.StandardError.ReadToEndAsync()
    }
    $Process.WaitForExit()

    if ($Redirect) {
        $script:NativeStdOut = $OutTask.Result
        $script:NativeStdErr = $ErrTask.Result
        if ($LogFile) {
            Add-Content -LiteralPath $LogFile -Value $script:NativeStdOut
            Add-Content -LiteralPath $LogFile -Value $script:NativeStdErr
        }
    } else {
        $script:NativeStdOut = ""
        $script:NativeStdErr = ""
    }
    return $Process.ExitCode
}

function Test-DockerArguments {
    param([string[]]$Arguments)
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        return $false
    }
    return (Invoke-NativeCommand docker $Arguments -Capture) -eq 0
}

function Find-Compose {
    if (Test-DockerArguments @("compose", "version")) {
        $script:ComposeStyle = "plugin"
        return $true
    }
    if ((Get-Command docker-compose -ErrorAction SilentlyContinue) -and
        (Invoke-NativeCommand docker-compose @("version") -Capture) -eq 0) {
        $script:ComposeStyle = "standalone"
        return $true
    }
    return $false
}

function Request-Retry {
    $Answer = Read-MenuChoice `
        "Press Enter to check again, or type q to exit" "q"
    if ($Answer -match '^[qQ]$') {
        throw "Canceled by user."
    }
}

function Confirm-Choice {
    param([string]$Prompt)
    $Answer = Read-MenuChoice "$Prompt [y/N]" "n"
    return $Answer -match '^(y|Y|yes|YES)$'
}

function Ensure-Docker {
    while (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host "Docker is required for this option."
        Write-Host "Installation guide: $DockerUrl"
        Request-Retry
    }
    while (-not (Test-DockerArguments @("info"))) {
        Write-Host "Docker is installed, but its service is not running."
        Write-Host "Start Docker Desktop, then return here."
        Request-Retry
    }
    while (-not (Find-Compose)) {
        Write-Host "Docker Compose is not available."
        Write-Host "Install the current Docker package from: $DockerUrl"
        Request-Retry
    }
}

function Invoke-Compose {
    param(
        [string[]]$Arguments,
        [switch]$Capture,
        [string]$LogFile
    )
    Push-Location $AppDir
    try {
        if ($ComposeStyle -eq "plugin") {
            return Invoke-NativeCommand docker `
                (@("compose") + $Arguments) -Capture:$Capture `
                -LogFile $LogFile
        }
        return Invoke-NativeCommand docker-compose $Arguments `
            -Capture:$Capture -LogFile $LogFile
    } finally {
        Pop-Location
    }
}

function Invoke-DatasetTool {
    param([string[]]$Arguments)
    Ensure-Docker
    New-Item -ItemType Directory -Force `
        -Path (Join-Path $RepoDir "datasets") | Out-Null
    $env:GEO_PARAMS_UID = "0"
    $env:GEO_PARAMS_GID = "0"
    Write-Host -NoNewline "Preparing the dataset manager... "
    $ExitCode = Invoke-Compose @("build", "dataset-manager") -Capture
    if ($ExitCode -ne 0) {
        Write-Host "failed"
        Write-Host $NativeStdErr
        return $ExitCode
    }
    Write-Host "done"
    return Invoke-Compose (@("run", "--rm", "dataset-manager") + $Arguments)
}

function Wait-ForUser {
    [void](Read-Host "Press Enter to continue")
}

function Read-MenuChoice {
    param([string]$Prompt, [string]$Default)
    $Value = Read-Host $Prompt
    if ($null -eq $Value) {
        return $Default
    }
    return $Value.Trim()
}
