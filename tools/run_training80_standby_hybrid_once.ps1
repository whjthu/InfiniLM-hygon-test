param(
    [ValidatePattern('^f[0-9]{2}r[0-9]+n[0-9]{2}$')]
    [string]$Node,
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$original = Join-Path $workspace 'run-dcu-standby-cycle-20261123.ps1'
$expectedHash = '04f2bef937f98b35a29c0bfce7c199e56e049e001a0a5b92ecbf12878a358c73'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $original).Hash.ToLowerInvariant() -ne $expectedHash) {
    throw 'The original standby launcher changed'
}
$source = [IO.File]::ReadAllText($original).Replace("`r`n", "`n")
$embedded = [regex]::Match($source, "(?ms)^@'\n(?<python>.*?)\n'@ \| ")
if (-not $embedded.Success) {
    throw 'The original Python launcher boundary changed'
}
$entry = $embedded.Groups['python'].Value
$oldBranch = "if via_compute:`n    probe_program="
$newBranch = "admission_via_compute = False`nif admission_via_compute:`n    probe_program="
if ([regex]::Matches($entry, [regex]::Escape($oldBranch)).Count -ne 1) {
    throw 'The original admission branch changed'
}
$entry = $entry.Replace($oldBranch, $newBranch)
$python = 'C:/Users/WangHaojie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$OutputEncoding = [Text.UTF8Encoding]::new($false)

if ($VerifyOnly) {
    $entry | & $python -B -X utf8 -c 'import ast,json,sys; tree=ast.parse(sys.stdin.read()); verify={n.test.id for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name)}; assert {"admission_via_compute","via_compute"}<=verify; print(json.dumps(dict(direct_admission=True, compute_bridge_dispatch=True, network_operations=0)))'
    exit $LASTEXITCODE
}
if (-not $Node) {
    throw 'Supply one node after reconciling the previous request and Slurm step'
}

$oldNodes = $env:HYGON_STANDBY_ONLY_NODES
$oldBridge = $env:HYGON_STANDBY_VIA_COMPUTE
$env:HYGON_STANDBY_ONLY_NODES = $Node
$env:HYGON_STANDBY_VIA_COMPUTE = '1'
Push-Location -LiteralPath $workspace
try {
    $entry | & $python -B -X utf8 -
    $result = $LASTEXITCODE
} finally {
    Pop-Location
    $env:HYGON_STANDBY_ONLY_NODES = $oldNodes
    $env:HYGON_STANDBY_VIA_COMPUTE = $oldBridge
}
exit $result
