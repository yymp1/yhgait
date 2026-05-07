[CmdletBinding()]
param(
    [ValidateSet("smoke", "probe", "short-run", "baseline", "formal-train", "formal-eval", "build-gallery")]
    [string]$Mode = "smoke",
    [string]$VenvDir = ".venvs\win11-demo",
    [int]$MasterPort = 29531,
    [int]$NumWorkers = 0,
    [int]$CheckpointIter = 1000,
    [int]$ResumeIter = 1000,
    [string]$SmokeConfig = "configs\opengait_casiab_smoke.yaml",
    [string]$BaselineConfig = "configs\opengait_casiab_baseline_small.yaml",
    [string]$FormalTrainConfig = "configs\opengait_casiab_formal_conservative.yaml",
    [string]$FormalEvalConfig = "configs\opengait_casiab_formal_conservative_eval.yaml",
    [string]$CachePath = "reports\gait_demo_cache_formal_conservative_real_iter01000.npz"
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

function Resolve-AbsolutePath([string]$BaseDir, [string]$Candidate) {
    if ([System.IO.Path]::IsPathRooted($Candidate)) {
        return [System.IO.Path]::GetFullPath($Candidate)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $Candidate))
}

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-AbsolutePath -BaseDir $ScriptRoot -Candidate ".."
$PythonExe = Join-Path (Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $VenvDir) "Scripts\python.exe"
$EnvCheckScript = Join-Path $RepoRoot "scripts\check_windows_demo_env.py"
$SmokeScript = Join-Path $RepoRoot "scripts\run_opengait_smoke_test.py"
$ProbeScript = Join-Path $RepoRoot "scripts\run_opengait_gpu_probe.py"
$MainScript = Join-Path $RepoRoot "scripts\run_opengait_main.py"
$GalleryScript = Join-Path $RepoRoot "scripts\build_gait_gallery.py"
$OpengaitRoot = Join-Path $RepoRoot "external\OpenGait"
$EnvCheckJson = Join-Path $RepoRoot "reports\windows_training_env_check.json"

if (-not (Test-Path $PythonExe)) {
    throw "未找到训练环境：$PythonExe 。先运行 scripts/start_win11_demo.ps1 -SkipLaunch。"
}

Write-Step "执行 Win11 训练环境检查"
Invoke-ExternalCommand -Exe $PythonExe -Args @(
    $EnvCheckScript,
    "--cfg-path",
    (Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $FormalEvalConfig),
    "--write-json",
    $EnvCheckJson,
    "--strict"
) -WorkingDirectory $RepoRoot

switch ($Mode) {
    "smoke" {
        Write-Step "运行 OpenGait 训练 smoke test"
        Invoke-ExternalCommand -Exe $PythonExe -Args @(
            $SmokeScript,
            "--python-bin",
            $PythonExe,
            "--opengait-root",
            $OpengaitRoot,
            "--config-path",
            (Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $SmokeConfig),
            "--master-port",
            $MasterPort.ToString()
        ) -WorkingDirectory $RepoRoot
    }
    "probe" {
        Write-Step "运行 OpenGait 单卡 probe"
        Invoke-ExternalCommand -Exe $PythonExe -Args @(
            $ProbeScript,
            "--python-bin",
            $PythonExe,
            "--opengait-root",
            $OpengaitRoot,
            "--config-path",
            (Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $BaselineConfig),
            "--mode",
            "probe",
            "--num-workers",
            $NumWorkers.ToString(),
            "--master-port",
            $MasterPort.ToString()
        ) -WorkingDirectory $RepoRoot
    }
    "short-run" {
        Write-Step "运行 OpenGait 单卡 short-run"
        Invoke-ExternalCommand -Exe $PythonExe -Args @(
            $ProbeScript,
            "--python-bin",
            $PythonExe,
            "--opengait-root",
            $OpengaitRoot,
            "--config-path",
            (Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $BaselineConfig),
            "--mode",
            "short-run",
            "--num-workers",
            $NumWorkers.ToString(),
            "--master-port",
            $MasterPort.ToString()
        ) -WorkingDirectory $RepoRoot
    }
    "baseline" {
        Write-Step "运行 OpenGait baseline small"
        Invoke-ExternalCommand -Exe $PythonExe -Args @(
            $ProbeScript,
            "--python-bin",
            $PythonExe,
            "--opengait-root",
            $OpengaitRoot,
            "--config-path",
            (Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $BaselineConfig),
            "--mode",
            "baseline",
            "--num-workers",
            $NumWorkers.ToString(),
            "--master-port",
            $MasterPort.ToString()
        ) -WorkingDirectory $RepoRoot
    }
    "formal-train" {
        Write-Step "启动 formal_conservative_real 单卡训练"
        $args = @(
            $MainScript,
            "--cfg-path",
            (Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $FormalTrainConfig),
            "--phase",
            "train",
            "--opengait-root",
            $OpengaitRoot,
            "--master-port",
            $MasterPort.ToString(),
            "--num-workers",
            $NumWorkers.ToString(),
            "--log-to-file"
        )
        if ($ResumeIter -ge 0) {
            $args += @("--iter", $ResumeIter.ToString())
        }
        Invoke-ExternalCommand -Exe $PythonExe -Args $args -WorkingDirectory $RepoRoot
    }
    "formal-eval" {
        Write-Step "启动 formal_conservative_real 单卡评估"
        Invoke-ExternalCommand -Exe $PythonExe -Args @(
            $MainScript,
            "--cfg-path",
            (Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $FormalEvalConfig),
            "--phase",
            "test",
            "--opengait-root",
            $OpengaitRoot,
            "--master-port",
            $MasterPort.ToString(),
            "--iter",
            $CheckpointIter.ToString(),
            "--num-workers",
            $NumWorkers.ToString(),
            "--log-to-file"
        ) -WorkingDirectory $RepoRoot
    }
    "build-gallery" {
        Write-Step "构建本地 gallery 缓存"
        Invoke-ExternalCommand -Exe $PythonExe -Args @(
            $GalleryScript,
            "--cfg-path",
            (Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $FormalEvalConfig),
            "--checkpoint-iter",
            $CheckpointIter.ToString(),
            "--cache-path",
            (Resolve-AbsolutePath -BaseDir $RepoRoot -Candidate $CachePath),
            "--master-port",
            $MasterPort.ToString(),
            "--log-to-file"
        ) -WorkingDirectory $RepoRoot
    }
}

Write-Host ""
Write-Host "训练动作已完成。环境检查报告：$EnvCheckJson" -ForegroundColor Green
