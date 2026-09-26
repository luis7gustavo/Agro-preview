param(
    [string]$OutputDirectory = "artifacts/release",
    [string]$BaseName = "agro-project-data-2026-09-26"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$output = Join-Path $projectRoot $OutputDirectory
New-Item -ItemType Directory -Path $output -Force | Out-Null
$archive = Join-Path $output "$BaseName.tar.gz"
$checksumPath = "$archive.sha256"
$manifestPath = Join-Path $output "$BaseName.manifest.json"
$staging = Join-Path $output ("staging-" + [guid]::NewGuid())
$targets = @(
    "data/bronze/era5",
    "data/silver",
    "data/gold",
    "models",
    "reports/generated",
    "mlruns"
)

foreach ($target in $targets) {
    if (-not (Test-Path -LiteralPath (Join-Path $projectRoot $target))) {
        throw "Diretorio obrigatorio ausente: $target"
    }
}

$entries = foreach ($target in $targets) {
    $files = @(Get-ChildItem -LiteralPath (Join-Path $projectRoot $target) -Recurse -File -Force)
    [ordered]@{
        path = $target
        files = $files.Count
        bytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
    }
}
$latest = Get-Content -LiteralPath (Join-Path $projectRoot "models/yield/soja/latest.json") -Raw |
    ConvertFrom-Json
$manifest = [ordered]@{
    schema_version = 1
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    archive = "$BaseName.tar.gz"
    included = $entries
    excluded = @(
        "data/bronze/inmet",
        "data/bronze/ibge",
        "data/bronze/mapa",
        "data/bronze/conab",
        "data/external",
        ".venv",
        "credentials",
        "caches"
    )
    promoted_model = $latest.model_version
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8

New-Item -ItemType Directory -Path $staging | Out-Null
try {
    foreach ($target in $targets) {
        $destination = Join-Path $staging $target
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $projectRoot $target) -Destination $destination -Recurse
    }

    # Acquisition manifests retain source checksums and relative artifact paths,
    # but their original machine-specific cache root is not portable or public.
    $era5Root = Join-Path $staging "data/bronze/era5"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    Get-ChildItem -LiteralPath $era5Root -Recurse -Filter "*.manifest.json" | ForEach-Object {
        $payload = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
        if ($null -eq $payload.source_artifact_root) {
            return
        }
        $payload.source_artifact_root = "data/bronze/era5/arco_raw"
        $payload | Add-Member -NotePropertyName release_path_normalization -NotePropertyValue (
            "portable relative cache root; source data bytes and checksums unchanged"
        ) -Force
        $json = $payload | ConvertTo-Json -Depth 20
        [System.IO.File]::WriteAllText($_.FullName, $json, $utf8NoBom)
    }

    Push-Location $staging
    & tar.exe -czf $archive @targets
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao criar o arquivo de dados."
    }
}
finally {
    if ((Get-Location).Path -eq $staging) {
        Pop-Location
    }
    $resolvedStaging = [System.IO.Path]::GetFullPath($staging)
    $resolvedOutput = [System.IO.Path]::GetFullPath($output) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $resolvedStaging.StartsWith($resolvedOutput) -or
        -not (Split-Path -Leaf $resolvedStaging).StartsWith("staging-")) {
        throw "Recusa ao limpar um diretorio temporario fora do escopo esperado."
    }
    Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
}

$hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash  $BaseName.tar.gz" | Set-Content -LiteralPath $checksumPath -Encoding ascii
Write-Output $archive
Write-Output $checksumPath
Write-Output $manifestPath
