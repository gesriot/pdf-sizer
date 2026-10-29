<#
.SYNOPSIS
Builds the Windows desktop executable with the app icon embedded.

.DESCRIPTION
The egui runtime icon is enough for the window, but Windows Explorer, pinned
shortcuts, and the taskbar rely on the executable's PE icon resources. This
script converts the existing PNG artwork to a multi-size ICO, compiles it into
a Windows resource with rc.exe, links that resource into the release binary,
and copies the packaged executable to dist/windows.

.PARAMETER SkipBuild
Skip Cargo build and only copy an already built executable.

.PARAMETER NoZip
Skip creation of dist/pdf-sizer-windows.zip.

.PARAMETER Target
Optional Rust target triple, for example x86_64-pc-windows-msvc.

Environment overrides:
  APP_NAME=pdf-sizer
  BINARY_NAME=pdf-sizer
  ICON_PNG=macos/icon-runtime.png
  DIST_DIR=dist/windows
#>

[CmdletBinding()]
param(
    [switch]$SkipBuild,
    [switch]$NoZip,
    [string]$Target,
    [string]$AppName,
    [string]$BinaryName,
    [string]$IconPng,
    [string]$DistDir
)

$ErrorActionPreference = "Stop"

function Get-Setting {
    param(
        [string]$Name,
        [string]$Default
    )

    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }
    return $value
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    Push-Location $WorkingDirectory
    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$FilePath failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

function Get-RustHostTriple {
    $rustc = Get-Command rustc -ErrorAction SilentlyContinue
    if (-not $rustc) {
        throw "rustc was not found in PATH."
    }

    $version = & $rustc.Source -vV
    foreach ($line in $version) {
        if ($line -match "^host:\s+(.+)$") {
            return $Matches[1]
        }
    }

    throw "Could not determine rustc host triple."
}

function Get-RcArch {
    param([string]$Triple)

    if ($Triple -match "aarch64|arm64") {
        return "arm64"
    }
    if ($Triple -match "x86_64|amd64") {
        return "x64"
    }
    if ($Triple -match "i686|i586|x86") {
        return "x86"
    }

    return "x64"
}

function Find-RcExe {
    param([string]$Arch)

    $cmd = Get-Command rc.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        return $cmd.Source
    }

    $roots = @()
    $programFilesX86 = [Environment]::GetEnvironmentVariable("ProgramFiles(x86)")
    $programFiles = [Environment]::GetEnvironmentVariable("ProgramFiles")
    if ($programFilesX86) {
        $roots += Join-Path $programFilesX86 "Windows Kits\10\bin"
    }
    if ($programFiles) {
        $roots += Join-Path $programFiles "Windows Kits\10\bin"
    }

    $archOrder = @($Arch, "x64", "x86", "arm64") | Select-Object -Unique
    foreach ($root in $roots) {
        if (-not (Test-Path -LiteralPath $root)) {
            continue
        }

        $versionDirs = Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending
        foreach ($versionDir in $versionDirs) {
            foreach ($candidateArch in $archOrder) {
                $candidate = Join-Path $versionDir.FullName (Join-Path $candidateArch "rc.exe")
                if (Test-Path -LiteralPath $candidate) {
                    return $candidate
                }
            }
        }
    }

    throw "rc.exe was not found. Install the Windows SDK or run this from a Visual Studio Developer PowerShell."
}

function Convert-PngToWindowsIcons {
    param(
        [string]$SourcePng,
        [string]$DestinationIco,
        [string]$DestinationPngDir
    )

    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "python was not found in PATH. Python with Pillow is required to create the Windows .ico file."
    }

    $pythonCode = @'
import sys
from pathlib import Path

try:
    from PIL import Image
except ModuleNotFoundError:
    print("error: Python Pillow package is required to convert PNG to ICO", file=sys.stderr)
    sys.exit(1)

src = Path(sys.argv[1])
ico_dst = Path(sys.argv[2])
png_dir = Path(sys.argv[3])
if not src.is_file():
    print(f"error: source icon not found: {src}", file=sys.stderr)
    sys.exit(1)

image = Image.open(src).convert("RGBA")
side = max(image.size)
canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
canvas.alpha_composite(image, ((side - image.width) // 2, (side - image.height) // 2))

sizes = [16, 24, 32, 48, 64, 128, 256]
ico_dst.parent.mkdir(parents=True, exist_ok=True)
png_dir.mkdir(parents=True, exist_ok=True)

for size in sizes:
    resized = canvas.resize((size, size), Image.Resampling.LANCZOS)
    resized.save(png_dir / f"icon-{size}.png")

canvas.save(
    ico_dst,
    format="ICO",
    sizes=[(size, size) for size in sizes],
)
'@

    $pythonCode | & $python.Source - $SourcePng $DestinationIco $DestinationPngDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to convert PNG icon to Windows icons."
    }
}

$rootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$rootPath = $rootDir.Path

if (-not $AppName) {
    $AppName = Get-Setting "APP_NAME" "pdf-sizer"
}
if (-not $BinaryName) {
    $BinaryName = Get-Setting "BINARY_NAME" "pdf-sizer"
}
if (-not $IconPng) {
    $IconPng = Get-Setting "ICON_PNG" (Join-Path $rootPath "macos\icon-runtime.png")
}
if (-not $DistDir) {
    $DistDir = Get-Setting "DIST_DIR" (Join-Path $rootPath "dist\windows")
}

if (-not [System.IO.Path]::IsPathRooted($IconPng)) {
    $IconPng = Join-Path $rootPath $IconPng
}
if (-not [System.IO.Path]::IsPathRooted($DistDir)) {
    $DistDir = Join-Path $rootPath $DistDir
}

$targetTriple = if ($Target) { $Target } else { Get-RustHostTriple }
if ($targetTriple -notmatch "windows") {
    throw "This packaging script can only build Windows targets. Target was: $targetTriple"
}
if ($targetTriple -notmatch "msvc") {
    throw "This packaging script currently supports windows-msvc targets. Target was: $targetTriple"
}

$workDir = Join-Path $rootPath "target\windows-package"
$iconIco = Join-Path $workDir "app.ico"
$iconPngDir = Join-Path $workDir "icons"
$resourceRc = Join-Path $workDir "app.rc"
$resourceRes = Join-Path $workDir "app.res"

New-Item -ItemType Directory -Force -Path $workDir | Out-Null
Convert-PngToWindowsIcons -SourcePng $IconPng -DestinationIco $iconIco -DestinationPngDir $iconPngDir

Set-Content -LiteralPath $resourceRc -Encoding ASCII -Value '1 ICON "app.ico"'
$rcExe = Find-RcExe -Arch (Get-RcArch $targetTriple)
Invoke-Checked -FilePath $rcExe -Arguments @("/nologo", "/fo$resourceRes", $resourceRc) -WorkingDirectory $workDir

if (-not $SkipBuild) {
    $cargoArgs = @("rustc", "--release", "--bin", $BinaryName)
    if ($Target) {
        $cargoArgs += @("--target", $Target)
    }
    $cargoArgs += @("--", "-C", "link-arg=$resourceRes")
    Invoke-Checked -FilePath "cargo" -Arguments $cargoArgs -WorkingDirectory $rootPath
}

$targetDir = if ($Target) {
    Join-Path $rootPath (Join-Path "target" (Join-Path $Target "release"))
} else {
    Join-Path $rootPath "target\release"
}
$builtExe = Join-Path $targetDir "$BinaryName.exe"
if (-not (Test-Path -LiteralPath $builtExe)) {
    throw "Built executable not found: $builtExe"
}

New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
$outExe = Join-Path $DistDir "$AppName.exe"
Copy-Item -LiteralPath $builtExe -Destination $outExe -Force
$outIco = Join-Path $DistDir "$AppName.ico"
Copy-Item -LiteralPath $iconIco -Destination $outIco -Force

$license = Join-Path $rootPath "LICENSE"
if (Test-Path -LiteralPath $license) {
    Copy-Item -LiteralPath $license -Destination (Join-Path $DistDir "LICENSE") -Force
}

if (-not $NoZip) {
    $zipPath = Join-Path (Split-Path -Parent $DistDir) "$AppName-windows.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $DistDir "*") -DestinationPath $zipPath -Force
}

Write-Host "Exe: $outExe"
Write-Host "ICO: $outIco"
Write-Host "PNGs: $iconPngDir"
if (-not $NoZip) {
    Write-Host "Zip: $zipPath"
}
