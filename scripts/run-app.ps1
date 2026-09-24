$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0

$GeoParamsScriptDir = $PSScriptRoot
. (Join-Path $GeoParamsScriptDir "lib\common.ps1")

$Port = if ($env:GEO_PARAMS_PORT) { $env:GEO_PARAMS_PORT } else { "8181" }
$LogDir = Join-Path $AppDir "log"
$LogFile = Join-Path $LogDir "docker-run.log"
$env:GEO_PARAMS_PORT = $Port

function Invoke-ComposeStep {
    param([string]$Label, [string[]]$Arguments)
    Write-Host -NoNewline "$Label... "
    $ExitCode = Invoke-Compose $Arguments -LogFile $LogFile
    if ($ExitCode -ne 0) {
        Write-Host "failed"
        Write-Host "See geo_params_web/log/docker-run.log for details."
        throw "$Label failed."
    }
    Write-Host "done"
}

Write-Host "GeoParams Web - guided startup"
Write-Host
Ensure-Docker

$PortNumber = 0
if (-not [int]::TryParse($Port, [ref]$PortNumber) -or
    $PortNumber -lt 1 -or $PortNumber -gt 65535) {
    throw "GEO_PARAMS_PORT must be a number between 1 and 65535."
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
    Write-Host "The application image dataset is not available yet."
    Write-Host "Use Dataset management in GeoParams to download it."
    Write-Host "Manual download: $DatasetUrl"
    Request-Retry
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

Invoke-ComposeStep "[1/4] Building the application" @(
    "build", "app"
)
if (Test-Path -LiteralPath $DatasetDir -PathType Container) {
    Invoke-ComposeStep "[2/4] Preparing petrographic images" @(
        "run", "--rm", "prepare"
    )
} else {
    Write-Host "[2/4] Using the existing image cache... done"
}
Invoke-ComposeStep "[3/4] Starting the web application" @(
    "up", "-d", "app", "nginx"
)

Write-Host -NoNewline "[4/4] Waiting for the web interface... "
$Ready = $false
for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
    $Health = Invoke-Compose @(
        "exec", "-T", "nginx", "wget", "-qO-",
        "http://127.0.0.1/health"
    ) -Capture
    $Application = Invoke-Compose @(
        "exec", "-T", "nginx", "wget", "-qO-",
        "http://127.0.0.1/geo-server/"
    ) -Capture
    if ($Health -eq 0 -and $Application -eq 0) {
        $Ready = $true
        break
    }
    Start-Sleep -Seconds 2
}

if (-not $Ready) {
    Write-Host "failed"
    [void](Invoke-Compose @("ps") -LogFile $LogFile)
    [void](Invoke-Compose @(
        "logs", "--no-color", "--tail=200"
    ) -LogFile $LogFile)
    throw "The application did not become ready. See $LogFile"
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
$Choice = Read-MenuChoice "Choice [1]" "1"

if ($Choice -eq "2") {
    Invoke-ComposeStep "Stopping the application" @("stop")
    Write-Host "The application is stopped. Its saved data was preserved."
} else {
    Write-Host "The application will keep running in the background."
}
Write-Host "Use GeoParams again to stop or remove its Docker resources."
