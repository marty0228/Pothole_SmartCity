Param(
    [string]$Url = "",
    [string]$Out = "weights/best.pt"
)

if (-not $Url) {
    Write-Host "다운로드 URL을 지정하세요. 예: .\scripts\download_weights.ps1 -Url 'https://example.com/best.pt'"
    exit 1
}

$dir = Split-Path $Out
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }

Write-Host "Downloading $Url -> $Out"
Invoke-WebRequest -Uri $Url -OutFile $Out
Write-Host "Done."
