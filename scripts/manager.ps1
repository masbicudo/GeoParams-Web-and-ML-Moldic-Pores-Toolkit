$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$GeoParamsScriptDir = $PSScriptRoot
. (Join-Path $GeoParamsScriptDir "lib\common.ps1")

function Show-CommandStatus {
    param([string]$Label, [string]$Command)
    if (Get-Command $Command -ErrorAction SilentlyContinue) {
        Write-Host "[OK]      $Label"
        return $true
    }
    Write-Host "[MISSING] $Label"
    return $false
}

$PythonCommand = $null
$PythonPrefix = @()

function Test-Python312 {
    $Candidates = @(
        @{ Command = "python"; Prefix = @() },
        @{ Command = "py"; Prefix = @("-3.12") }
    )
    foreach ($Candidate in $Candidates) {
        $Found = Get-Command $Candidate.Command -ErrorAction SilentlyContinue
        if (-not $Found) {
            continue
        }
        $Arguments = @($Candidate.Prefix) + @(
            "-c",
            "import sys; raise SystemExit(sys.version_info[:2] != (3, 12))"
        )
        if ((Invoke-NativeCommand $Found.Source $Arguments -Capture) -eq 0) {
            $script:PythonCommand = $Found.Source
            $script:PythonPrefix = @($Candidate.Prefix)
            Write-Host "[OK]      Python 3.12"
            return $true
        }
    }
    Write-Host "[MISSING] Python 3.12"
    return $false
}

function Test-AppRuntime {
    Write-Host "Requirements: run the web application"
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        Write-Host "[FOUND]   Docker command"
        if (Test-DockerArguments @("info")) {
            Write-Host "[OK]      Docker service is running"
        } else {
            Write-Host "[ACTION]  Start Docker Desktop"
        }
        if (Find-Compose) {
            Write-Host "[OK]      Docker Compose is available"
        } else {
            Write-Host "[MISSING] Docker Compose"
        }
    } else {
        Write-Host "[MISSING] Docker"
        Write-Host "          $DockerUrl"
    }
    Write-Host "Python and PDM are not required for the Docker application."
}

function Test-AppDevelopment {
    Write-Host "Requirements: develop the web application"
    if (-not (Test-Python312)) {
        Write-Host "          https://www.python.org/downloads/"
    }
    if (-not (Show-CommandStatus "PDM" "pdm")) {
        Write-Host "          https://pdm-project.org/latest/#installation"
    }
    [void](Show-CommandStatus "Docker (recommended for parity)" "docker")
}

function Test-Models {
    param([string]$Mode)
    Write-Host "Requirements: $Mode models and research analyses"
    if (-not (Test-Python312)) {
        Write-Host "          https://www.python.org/downloads/"
    }
    if (-not (Show-CommandStatus "PDM" "pdm")) {
        Write-Host "          https://pdm-project.org/latest/#installation"
    }
    $DatasetPath = Join-Path $RepoDir "datasets\pore_type_training"
    if (Test-Path -LiteralPath $DatasetPath -PathType Container) {
        Write-Host "[FOUND]   Model datasets directory"
    } else {
        Write-Host "[MISSING] Model datasets; use Dataset management"
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        Write-Host "[OK]      NVIDIA GPU tools"
    } else {
        Write-Host "[OPTIONAL] NVIDIA GPU tools for accelerated workflows"
    }
    if ($Mode -eq "develop") {
        [void](Show-CommandStatus "Visual Studio Code (optional)" "code")
    }
}

function Invoke-SmokeTests {
    if (-not (Test-Python312)) {
        Write-Host "Install Python 3.12 manually:"
        Write-Host "https://www.python.org/downloads/"
        return
    }
    Push-Location $RepoDir
    try {
        [void](Invoke-NativeCommand $PythonCommand `
            ($PythonPrefix + @("smoke_test_notebooks.py")))
    } finally {
        Pop-Location
    }
}

function Show-RequirementsMenu {
    while ($true) {
        Write-Host
        Write-Host "Requirements and tests"
        Write-Host "  1) Check requirements to run the web application"
        Write-Host "  2) Check requirements to develop the web application"
        Write-Host "  3) Check requirements to run models and analyses"
        Write-Host "  4) Check requirements to develop models"
        Write-Host "  5) Run the safe notebook smoke test"
        Write-Host "  b) Back"
        $Choice = Read-MenuChoice "Choice" "b"
        switch ($Choice) {
            "1" { Test-AppRuntime }
            "2" { Test-AppDevelopment }
            "3" { Test-Models "run" }
            "4" { Test-Models "develop" }
            "5" { Invoke-SmokeTests }
            { $_ -match '^[bB]$' } { return }
            default { Write-Host "Invalid choice." }
        }
        Wait-ForUser
    }
}

function Invoke-DatasetDownload {
    param([string]$Scope, [switch]$ReplaceInvalid)
    $Size = if ($Scope -eq "app") { "about 329 MB" } else { "about 1.14 GB" }
    Write-Host "This will download $Size into: $(Join-Path $RepoDir 'datasets')"
    Write-Host "Every file is accepted only after SHA-256 verification."
    if (-not (Confirm-Choice "Continue with this download?")) {
        Write-Host "Download canceled."
        return
    }
    $Arguments = @("download", "--scope", $Scope)
    if ($ReplaceInvalid) {
        $Arguments += "--replace-invalid"
    }
    [void](Invoke-DatasetTool $Arguments)
}

function Show-DatasetsMenu {
    while ($true) {
        Write-Host
        Write-Host "Dataset management"
        Write-Host "  1) Verify application datasets"
        Write-Host "  2) Download missing application datasets (~329 MB)"
        Write-Host "  3) Verify all research datasets"
        Write-Host "  4) Download all missing research datasets (~1.14 GB)"
        Write-Host "  5) Preserve and replace invalid files"
        Write-Host "  6) Show the manual Google Drive link"
        Write-Host "  b) Back"
        $Choice = Read-MenuChoice "Choice" "b"
        switch ($Choice) {
            "1" { [void](Invoke-DatasetTool @("verify", "--scope", "app")) }
            "2" { Invoke-DatasetDownload "app" }
            "3" { [void](Invoke-DatasetTool @("verify", "--scope", "all")) }
            "4" { Invoke-DatasetDownload "all" }
            "5" {
                Write-Host "Invalid files are renamed, never silently deleted."
                Invoke-DatasetDownload "all" -ReplaceInvalid
            }
            "6" {
                Write-Host $DatasetUrl
                Write-Host "Manual downloads do not require Docker."
            }
            { $_ -match '^[bB]$' } { return }
            default { Write-Host "Invalid choice." }
        }
        Wait-ForUser
    }
}

while ($true) {
    Write-Host
    Write-Host "GeoParams repository manager"
    Write-Host "  1) Run or install the web application with Docker"
    Write-Host "  2) Manage and verify public datasets"
    Write-Host "  3) Check requirements and run tests"
    Write-Host "  4) Stop or uninstall the Docker application"
    Write-Host "  q) Quit"
    $Choice = Read-MenuChoice "Choice" "q"
    switch ($Choice) {
        "1" { & (Join-Path $GeoParamsScriptDir "run-app.ps1") }
        "2" { Show-DatasetsMenu }
        "3" { Show-RequirementsMenu }
        "4" { & (Join-Path $GeoParamsScriptDir "remove-docker-app.ps1") }
        { $_ -match '^[qQ]$' } { exit 0 }
        default { Write-Host "Invalid choice." }
    }
}
