# Build weldpool_2d.mph for COMSOL Multiphysics 6.2 and run a short smoke solve.
# Requires Heat Transfer + CFD licenses. Class files must live under the COMSOL
# user application-files directory (security "Temporary and application files").

$ErrorActionPreference = "Stop"

$ComsolBin = "C:\Program Files\COMSOL\COMSOL62\Multiphysics\bin\win64"
$Compile = Join-Path $ComsolBin "comsolcompile.exe"
$Batch = Join-Path $ComsolBin "comsolbatch.exe"

$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$UserDir = Join-Path $env:USERPROFILE ".comsol\v62\applications\files\user"
$WorkDir = Join-Path $UserDir "weldpool_2d_build"

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType Directory -Force -Path $UserDir | Out-Null

Copy-Item (Join-Path $RepoDir "WeldPool2D.java") (Join-Path $WorkDir "WeldPool2D.java") -Force
Copy-Item (Join-Path $RepoDir "SmokeTest.java") (Join-Path $WorkDir "SmokeTest.java") -Force

Write-Host "== Compile WeldPool2D =="
& $Compile (Join-Path $WorkDir "WeldPool2D.java")
if (-not (Test-Path (Join-Path $WorkDir "WeldPool2D.class"))) {
  throw "comsolcompile WeldPool2D failed"
}

Write-Host "== Build MPH (geometry/mesh only; study not computed) =="
# Do not pass -norun: that skips run(). WeldPool2D.run() builds and saves without sol.runAll().
$buildLog = Join-Path $WorkDir "build.log"
& $Batch `
  -inputfile (Join-Path $WorkDir "WeldPool2D.class") `
  -outputfile (Join-Path $UserDir "weldpool_2d.mph") `
  -batchlog $buildLog
Get-Content $buildLog -Tail 40
$buildText = Get-Content $buildLog -Raw
if ($buildText -match "Error running java class" -or $buildText -match "Unknown parameter") {
  throw "MPH build failed; see $buildLog"
}

$mphUser = Join-Path $UserDir "weldpool_2d.mph"
if (-not (Test-Path $mphUser)) {
  throw "Expected MPH not found: $mphUser"
}

Copy-Item $mphUser (Join-Path $RepoDir "weldpool_2d.mph") -Force
Write-Host "Wrote $(Join-Path $RepoDir 'weldpool_2d.mph')"

Write-Host "== Compile SmokeTest =="
& $Compile (Join-Path $WorkDir "SmokeTest.java")
if (-not (Test-Path (Join-Path $WorkDir "SmokeTest.class"))) {
  throw "comsolcompile SmokeTest failed"
}

Write-Host "== Smoke solve (0.02 s, nosave) =="
$smokeLog = Join-Path $WorkDir "smoke.log"
& $Batch `
  -inputfile (Join-Path $WorkDir "SmokeTest.class") `
  -nosave `
  -batchlog $smokeLog
Get-Content $smokeLog -Tail 80
$smokeText = Get-Content $smokeLog -Raw
if ($smokeText -match "Error running java class" -or $smokeText -match "Error in multiphysics") {
  throw "smoke solve failed; see $smokeLog"
}

Write-Host "Done."
