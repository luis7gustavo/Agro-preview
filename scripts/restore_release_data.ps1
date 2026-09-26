param(
    [string]$Repository = "luis7gustavo/Agro-preview",
    [string]$Tag = "backup-2026-09-26",
    [string]$Asset = "agro-project-data-2026-09-26.tar.gz"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$temporary = Join-Path ([System.IO.Path]::GetTempPath()) ("agro-restore-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $temporary | Out-Null

try {
    $archive = Join-Path $temporary $Asset
    $checksumFile = "$archive.sha256"
    $baseUrl = "https://github.com/$Repository/releases/download/$Tag"
    Invoke-WebRequest -Uri "$baseUrl/$Asset" -OutFile $archive
    Invoke-WebRequest -Uri "$baseUrl/$Asset.sha256" -OutFile $checksumFile

    $expected = ((Get-Content -LiteralPath $checksumFile -Raw).Trim() -split "\s+")[0].ToLowerInvariant()
    if ($expected -notmatch "^[0-9a-f]{64}$") {
        throw "O arquivo de checksum publicado e invalido."
    }
    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
        throw "SHA-256 divergente; o snapshot nao sera extraido."
    }

    & tar.exe -xzf $archive -C $projectRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao extrair o snapshot."
    }
    Write-Host "Snapshot restaurado e verificado em $projectRoot"
}
finally {
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Recurse -Force
    }
}
