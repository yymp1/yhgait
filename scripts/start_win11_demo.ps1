[CmdletBinding()]
param(
    [string]$PythonVersion = "3.11",
    [string]$VenvDir = ".venvs\win11-demo",
    [string]$CfgPath = "configs\opengait_casiab_formal_conservative_eval.yaml",
    [int]$CheckpointIter = 1000,
    [string]$PrivateGalleryRoot = "data\private_gallery",
    [string]$VideoRunRoot = "reports\private_gallery_runs",
    [string]$ServerName = "127.0.0.1",
    [int]$ServerPort = 7860,
    [string]$TorchVersion = "2.11.0",
    [string]$TorchvisionVersion = "0.26.0",
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu126",
    [switch]$UseCpuTorch,
    [switch]$RecreateVenv,
    [switch]$SkipOpenGaitClone,
    [switch]$SkipLaunch,
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Format-Command([string]$Exe, [string[]]$Args) {
    $pieces = @($Exe)
    foreach ($arg in $Args) {
        if ($arg -match "\s") {
            $pieces += '"' + $arg + '"'
        }
        else {
            $pieces += $arg
        }
    }
    return ($pieces -join " ")
}

function Invoke-ExternalCommand([string]$Exe, [string[]]$Args, [string]$WorkingDirectory = "") {
    Write-Host ('$ ' + (Format-Command -Exe $Exe -Args $Args)) -ForegroundColor DarkGray
    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
    }
    try {
        & $Exe @Args
        if ($LASTEXITCODE -ne 0) {
            throw "命令执行失败，退出码：$LASTEXITCODE"
        }
    }
    finally {
        if ($WorkingDirectory) {
            Pop-Location
        }
    }
}

function Get-PythonVersionString([string]$Exe, [string[]]$Args) {
    $versionArgs = @()
    if ($Args) {
        $versionArgs += $Args
    }
    $versionArgs += @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    try {
        $output = & $Exe @versionArgs 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        return ($output | Select-Object -First 1).Trim()
    }
    catch {
        return $null
    }
}

function Get-BootstrapPythonCommand([string]$RequestedVersion) {
    $acceptableVersions = @($RequestedVersion, "3.11", "3.10") | Select-Object -Unique
    $pyCommand = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCommand) {
        foreach ($version in $acceptableVersions) {
            try {
                & $pyCommand.Source "-$version" "--version" *> $null
                if ($LASTEXITCODE -eq 0) {
                    return @($pyCommand.Source, "-$version")
                }
            }
            catch {
            }
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $detectedVersion = Get-PythonVersionString -Exe $pythonCommand.Source -Args @()
        if ($acceptableVersions -contains $detectedVersion) {
            return @($pythonCommand.Source)
        }
        if ($detectedVersion) {
            throw "当前 python 命令是 $detectedVersion，推荐安装 Python 3.10 或 3.11。"
        }
    }

    throw "未找到可用 Python。请先安装 Python 3.11，并勾选 Add Python to PATH。"
}

function Resolve-AbsolutePath([string]$BaseDir, [string]$Candidate) {
    if ([System.IO.Path]::IsPathRooted($Candidate)) {
        return [System.IO.Path]::GetFullPath($Candidate)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $Candidate))
}

function Ensure-OpenGaitRepo([string]$RepoRoot, [bool]$AllowClone) {
    $openGaitRoot = Join-Path $RepoRoot "external\OpenGait"
    if (Test-Path $openGaitRoot) {
        return
    }

    if (-not $AllowClone) {
        throw "缺少 external\OpenGait。请先从旧机器复制，或重新运行脚本并允许自动 clone。"
    }

    $gitCommand = Get-Command git -ErrorAction SilentlyContinue
    if (-not $gitCommand) {
        throw "缺少 git，无法自动 clone OpenGait。请先安装 Git，或手动复制 external\OpenGait。"
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $openGaitRoot -Parent) | Out-Null
    Invoke-ExternalCommand -Exe $gitCommand.Source -Args @(
        "clone",
        "--depth",
        "1",
        "https://github.com/ShiqiYu/OpenGait.git",
        $openGaitRoot
    )
}

function Get-FreePort([int]$PreferredPort) {
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $PreferredPort)
        $listener.Start()
        $listener.Stop()
        return $PreferredPort
    }
    catch {
    }

    $fallback = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    $fallback.Start()
    $port = $fallback.LocalEndpoint.Port
    $fallback.Stop()
    return $port
}

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-AbsolutePath -BaseDir $ScriptRoot -Candidate ".."
$VenvPath = Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $VenvDir
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsPath = Join-Path $RepoRoot "requirements-gait-demo.txt"
$EnvCheckScript = Join-Path $RepoRoot "scripts\check_windows_demo_env.py"
$CfgAbsolutePath = Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $CfgPath
$GalleryAbsolutePath = Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $PrivateGalleryRoot
$VideoRunAbsolutePath = Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $VideoRunRoot
$ReportsDir = Join-Path $RepoRoot "reports"
$EnvCheckJson = Join-Path $ReportsDir "windows_env_check.json"
$AllowClone = -not $SkipOpenGaitClone.IsPresent

Write-Step "检查仓库和 OpenGait 目录"
if (-not (Test-Path $RepoRoot)) {
    throw "项目根目录不存在：$RepoRoot"
}
Ensure-OpenGaitRepo -RepoRoot $RepoRoot -AllowClone $AllowClone
New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
New-Item -ItemType Directory -Force -Path $GalleryAbsolutePath | Out-Null
New-Item -ItemType Directory -Force -Path $VideoRunAbsolutePath | Out-Null

Write-Step "检查显卡驱动可见性"
$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) {
    Invoke-ExternalCommand -Exe $nvidiaSmi.Source -Args @()
}
else {
    Write-Host "未找到 nvidia-smi，后续会继续通过 PyTorch 检查 CUDA。" -ForegroundColor Yellow
}

Write-Step "准备 Python 虚拟环境"
$bootstrap = Get-BootstrapPythonCommand -RequestedVersion $PythonVersion
if ($RecreateVenv.IsPresent -and (Test-Path $VenvPath)) {
    Remove-Item -Recurse -Force $VenvPath
}
if (-not (Test-Path $PythonExe)) {
    $createVenvArgs = @()
    if ($bootstrap.Length -gt 1) {
        $createVenvArgs += $bootstrap[1..($bootstrap.Length - 1)]
    }
    $createVenvArgs += @("-m", "venv", $VenvPath)
    Invoke-ExternalCommand -Exe $bootstrap[0] -Args $createVenvArgs
}

Write-Step "升级 pip / setuptools / wheel"
Invoke-ExternalCommand -Exe $PythonExe -Args @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

Write-Step "安装 PyTorch"
if ($UseCpuTorch.IsPresent) {
    $torchIndex = "https://download.pytorch.org/whl/cpu"
}
else {
    $torchIndex = $TorchIndexUrl
}
Invoke-ExternalCommand -Exe $PythonExe -Args @(
    "-m",
    "pip",
    "install",
    "torch==$TorchVersion",
    "torchvision==$TorchvisionVersion",
    "--index-url",
    $torchIndex
)

Write-Step "安装 GUI / OpenGait demo 依赖"
Invoke-ExternalCommand -Exe $PythonExe -Args @("-m", "pip", "install", "-r", $RequirementsPath)

Write-Step "执行本地环境检查"
Invoke-ExternalCommand -Exe $PythonExe -Args @(
    $EnvCheckScript,
    "--cfg-path",
    $CfgAbsolutePath,
    "--checkpoint-iter",
    $CheckpointIter.ToString(),
    "--gallery-root",
    $GalleryAbsolutePath,
    "--write-json",
    $EnvCheckJson,
    "--strict"
) -WorkingDirectory $RepoRoot

if ($SkipLaunch.IsPresent) {
    Write-Host ""
    Write-Host "环境准备完成，已跳过启动。环境检查报告：$EnvCheckJson" -ForegroundColor Green
    exit 0
}

$LaunchPort = Get-FreePort -PreferredPort $ServerPort
$Url = "http://$ServerName`:$LaunchPort"

Write-Step "启动 Gradio 私有步态库 demo"
Write-Host "浏览器地址：$Url" -ForegroundColor Green
Write-Host "环境检查报告：$EnvCheckJson" -ForegroundColor Green
if (-not $NoBrowser.IsPresent) {
    Start-Process powershell -ArgumentList @(
        "-NoProfile",
        "-Command",
        "Start-Sleep -Seconds 4; Start-Process '$Url'"
    ) | Out-Null
}

Invoke-ExternalCommand -Exe $PythonExe -Args @(
    "app\gradio_gait_demo.py",
    "--cfg-path",
    $CfgAbsolutePath,
    "--checkpoint-iter",
    $CheckpointIter.ToString(),
    "--private-gallery-root",
    $GalleryAbsolutePath,
    "--video-run-root",
    $VideoRunAbsolutePath,
    "--server-name",
    $ServerName,
    "--server-port",
    $LaunchPort.ToString()
) -WorkingDirectory $RepoRoot
