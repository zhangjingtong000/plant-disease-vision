# =====================================================================
#  run_all.ps1 — 按顺序跑完整实验流程
#
#  顺序：训练三模型 → 评估出图 → 性能基准 → 退化实验 → 遮挡实验
#  用法：
#      pip install -r requirements.txt
#      pwsh ./run_all.ps1            # 或 powershell -File run_all.ps1
#      $env:PYTHON="C:\Python\python.exe"; ./run_all.ps1   # 指定解释器
# =====================================================================

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ROOT = $PSScriptRoot
$PY   = if ($env:PYTHON) { $env:PYTHON } else { "python" }

Set-Location $ROOT

Write-Host "==== 1/5 训练三个模型 ====" -ForegroundColor Cyan
foreach ($m in @("resnet50", "efficientnet_b0", "mobilenet_v3_large")) {
    Write-Host "训练 $m ..." -ForegroundColor Yellow
    & $PY "src\train.py" --model $m --epochs 25
}

Write-Host "==== 2/5 评估出图 ====" -ForegroundColor Cyan
foreach ($m in @("resnet50", "efficientnet_b0", "mobilenet_v3_large")) {
    $ckpt = "outputs\checkpoints\${m}_best.pth"
    if (Test-Path $ckpt) {
        & $PY "src\evaluate.py" --ckpt $ckpt --tag $m
    }
}

Write-Host "==== 3/5 性能基准 ====" -ForegroundColor Cyan
& $PY "src\benchmark.py"

Write-Host "==== 4/5 退化鲁棒性 ====" -ForegroundColor Cyan
& $PY "src\degenerate.py" --ckpt "outputs\checkpoints\resnet50_best.pth" --tag resnet50

Write-Host "==== 5/5 遮挡实验 ====" -ForegroundColor Cyan
& $PY "src\occlusion.py" --ckpt "outputs\checkpoints\resnet50_best.pth" --tag resnet50

Write-Host ""
Write-Host "全部完成，产物在 outputs\ 目录" -ForegroundColor Green
