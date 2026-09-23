$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$AppDir = Join-Path $PSScriptRoot "geo_params_web"
$ProjectLabel = "io.geoparams.project=geo-params-web"
$ManagerLabel = "io.geoparams.managed-by=geo-params-launcher"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is unavailable. Start it before managing the app."
    exit 1
}
& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker is unavailable. Start it before managing the app."
    exit 1
}

Push-Location $AppDir
try {
    function Get-ProjectContainerIds {
        @(& docker container ls -aq `
            --filter "label=$ProjectLabel" `
            --filter "label=$ManagerLabel")
    }

    function Stop-ProjectContainers {
        $Ids = @(& docker container ls -q `
            --filter "label=$ProjectLabel" `
            --filter "label=$ManagerLabel")
        if ($Ids.Count -eq 0) {
            Write-Host "No running project containers were found."
        } else {
            & docker container stop $Ids
        }
    }

    function Remove-ProjectContainers {
        $Ids = @(Get-ProjectContainerIds)
        if ($Ids.Count -eq 0) {
            Write-Host "No project-labeled containers were found."
        } else {
            & docker container rm -f $Ids
        }
    }

    function Remove-ProjectNetworks {
        $Ids = @(& docker network ls -q `
            --filter "label=$ProjectLabel" `
            --filter "label=$ManagerLabel")
        if ($Ids.Count -eq 0) {
            Write-Host "No project-labeled networks were found."
        } else {
            & docker network rm $Ids
        }
    }

    Write-Host "GeoParams Web - Docker cleanup"
    Write-Host "  1) Stop the application and keep its containers"
    Write-Host "  2) Remove its containers and private network"
    Write-Host "  3) Also remove images built and labeled by this project"
    Write-Host "  q) Cancel"
    $Choice = Read-Host "Choice"

    switch ($Choice) {
        "1" {
            Stop-ProjectContainers
        }
        "2" {
            Remove-ProjectContainers
            Remove-ProjectNetworks
        }
        "3" {
            Remove-ProjectContainers
            Remove-ProjectNetworks
            $ImageIds = @(& docker image ls `
                --filter "label=$ProjectLabel" `
                --filter "label=$ManagerLabel" `
                --format "{{.ID}}") | Sort-Object -Unique
            if ($ImageIds.Count -eq 0) {
                Write-Host "No project-labeled images were found."
            } else {
                foreach ($ImageId in $ImageIds) {
                    & docker image rm $ImageId
                }
            }
        }
        { $_ -in @("q", "Q") } {
            Write-Host "Nothing was changed."
            exit 0
        }
        default {
            Write-Host "Invalid choice. Nothing was changed."
            exit 1
        }
    }
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    Write-Host "Saved uploads and results were not deleted."
} finally {
    Pop-Location
}
