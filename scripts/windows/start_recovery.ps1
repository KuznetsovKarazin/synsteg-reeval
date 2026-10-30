param(
    [ValidateSet('Prepare', 'Local', 'Claude', 'Audit', 'Status', 'Checkpoint')]
    [string]$Stage = 'Prepare',
    [string]$Python = 'py',
    [string]$Model = ''
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $ProjectRoot
$VenvPython = Join-Path $ProjectRoot '.venv-recovery\Scripts\python.exe'
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE : $Program" }
}
if ($Stage -eq 'Prepare') {
    if (-not (Test-Path $VenvPython)) {
        if ($Python -eq 'py') { Invoke-Checked -Program $Python -Arguments @('-3.12', '-m', 'venv', '.venv-recovery') }
        else { Invoke-Checked -Program $Python -Arguments @('-m', 'venv', '.venv-recovery') }
    }
    Invoke-Checked -Program $VenvPython -Arguments @('-m', 'pip', 'install', '-e', '.[test]', '-r', 'requirements-shared-dictionary.txt')
    Invoke-Checked -Program $VenvPython -Arguments @('scripts/run_recovery_campaign.py', 'prepare')
} else {
    if (-not (Test-Path $VenvPython)) { throw 'Run -Stage Prepare first (Python 3.12 required).' }
    $Arguments = @('scripts/run_recovery_campaign.py', $Stage.ToLower())
    if ($Model) { $Arguments += @('--model', $Model) }
    if ($Stage -eq 'Local') {
        Invoke-Checked -Program $VenvPython -Arguments @('-m', 'pip', 'install', 'torch>=2.6,<3', '--index-url', 'https://download.pytorch.org/whl/cpu')
        Invoke-Checked -Program $VenvPython -Arguments @('-m', 'pip', 'install', '-r', 'requirements-neural.txt')
        $CacheArgs = @('scripts/run_recovery_campaign.py', 'cache-models')
        if ($Model) { $CacheArgs += @('--model', $Model) }
        Invoke-Checked -Program $VenvPython -Arguments $CacheArgs
    }
    if ($Stage -in @('Local', 'Claude')) { $Arguments += '--execute' }
    Invoke-Checked -Program $VenvPython -Arguments $Arguments
}
