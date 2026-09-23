$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoDir = $PSScriptRoot
$AppDir = Join-Path $RepoDir "geo_params_web"
$InstallUrl = "https://docs.docker.com/get-started/get-docker/"
$DatasetUrl = "https://drive.google.com/drive/folders/1s-NAWbgukQG-1Q3M5MpO808XRqA1QVw4?usp=sharing"
$Port = if ($env:GEO_PARAMS_PORT) { $env:GEO_PARAMS_PORT } else { "8181" }
$Image = if ($env:GEO_PARAMS_IMAGE) {
    $env:GEO_PARAMS_IMAGE
} else {
    "geo-params-web-app:app-v2"
}
$LogDir = Join-Path $AppDir "log"
$LogFile = Join-Path $LogDir "docker-run.log"

$env:GEO_PARAMS_PORT = $Port
$env:GEO_PARAMS_IMAGE = $Image

function Request-Retry {
    $Answer = Read-Host "Press Enter to check again, or type q to exit"
    if ($Answer -eq "q") {
        exit 1
    }
}

function Test-DockerCommand {
    param([string[]]$Arguments)
    & docker @Arguments *> $null
    return $LASTEXITCODE -eq 0
}

function Invoke-DockerStep {
    param(
        [string]$Label,
        [string[]]$Arguments
    )
    Write-Host -NoNewline "$Label... "
    & docker @Arguments *>> $LogFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "failed"
        Write-Host "See geo_params_web/log/docker-run.log for details."
        exit 1
    }
    Write-Host "done"
}

Write-Host "GeoParams Web - guided startup"
Write-Host

while (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed or is unavailable in this terminal."
    Write-Host "Installation guide: $InstallUrl"
    Write-Host "Install Docker, then return here. This script will check again."
    Request-Retry
}

while (-not (Test-DockerCommand @("compose", "version"))) {
    Write-Host "Docker Compose is not available."
    Write-Host "Install the current Docker package from: $InstallUrl"
    Request-Retry
}

while (-not (Test-DockerCommand @("info"))) {
    Write-Host "Docker is installed, but its service is not running."
    Write-Host "Start Docker Desktop, then return here."
    Request-Retry
}

$DatasetDir = Join-Path $RepoDir "datasets\article_thin_sections"
$CacheDir = Join-Path $AppDir "static\imgs_sections"
$MetadataFile = Join-Path $CacheDir "metadata.json"
$CachedImages = @(Get-ChildItem -LiteralPath $CacheDir -File `
    -ErrorAction SilentlyContinue | Where-Object {
        $_.Extension -in @(".jpg", ".jpeg", ".png")
    })
$CacheReady = (Test-Path -LiteralPath $MetadataFile -PathType Leaf) -and
    $CachedImages.Count -gt 0
while (-not (Test-Path -LiteralPath $DatasetDir -PathType Container) -and
    -not $CacheReady) {
    Write-Host "The public petrographic image dataset was not found."
    Write-Host "Download it from: $DatasetUrl"
    Write-Host "Place it in: $DatasetDir"
    Request-Retry
}

$PortNumber = 0
if (-not [int]::TryParse($Port, [ref]$PortNumber) -or
    $PortNumber -lt 1 -or $PortNumber -gt 65535) {
    Write-Host "GEO_PARAMS_PORT must be a number between 1 and 65535."
    exit 1
}

@(
    (Join-Path $AppDir "static\output"),
    (Join-Path $AppDir "static\imgs_sections"),
    (Join-Path $AppDir "data\uploads"),
    $LogDir
) | ForEach-Object {
    New-Item -ItemType Directory -Force -Path $_ | Out-Null
}
Set-Content -LiteralPath $LogFile -Value ""

Push-Location $AppDir
try {
    Invoke-DockerStep "[1/4] Building the application" @(
        "compose", "build"
    )
    if (Test-Path -LiteralPath $DatasetDir -PathType Container) {
        Invoke-DockerStep "[2/4] Preparing petrographic images" @(
            "compose", "run", "--rm", "prepare"
        )
    } else {
        Write-Host "[2/4] Using the existing image cache... done"
    }
    Invoke-DockerStep "[3/4] Starting the web application" @(
        "compose", "up", "-d", "app", "nginx"
    )

    Write-Host -NoNewline "[4/4] Waiting for the web interface... "
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
        & docker compose exec -T nginx wget -qO- `
            http://127.0.0.1/health *> $null
        $HealthReady = $LASTEXITCODE -eq 0
        & docker compose exec -T nginx wget -qO- `
            http://127.0.0.1/geo-server/ *> $null
        if ($HealthReady -and $LASTEXITCODE -eq 0) {
            $Ready = $true
            break
        }
        Start-Sleep -Seconds 2
    }

    if (-not $Ready) {
        Write-Host "failed"
        & docker compose ps *>> $LogFile
        & docker compose logs --no-color --tail=200 *>> $LogFile
        Write-Host "The application did not become ready."
        Write-Host "See geo_params_web/log/docker-run.log for details."
        exit 1
    }
    Write-Host "done"

    $Url = "http://localhost:$Port/geo-server/"
    Write-Host
    Write-Host "GeoParams Web is ready: $Url"
    Write-Host "Uploads and results remain in geo_params_web/data/uploads/."
    Write-Host
    Write-Host "What would you like to do now?"
    Write-Host "  1) Keep the application running in the background"
    Write-Host "  2) Stop the application now"
    $Choice = Read-Host "Choice [1]"

    if ($Choice -eq "2") {
        Invoke-DockerStep "Stopping the application" @("compose", "stop")
        Write-Host "The application is stopped. Its saved data was preserved."
    } else {
        Write-Host "The application will keep running in the background."
    }

    Write-Host "Use remove-docker-app.bat to stop or remove Docker resources."
} finally {
    Pop-Location
}
