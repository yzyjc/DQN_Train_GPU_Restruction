[CmdletBinding()]
param(
    [int]$ViewerId = 0,
    [Parameter(Mandatory = $true)][string]$EventTimingCsv,
    [Parameter(Mandatory = $true)][string]$OutCsv
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Import-Module (Join-Path $PSScriptRoot 'rgp_uia_common.psm1') -Force
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class RgpShaderWindow {
    [DllImport("user32.dll", SetLastError=true)]
    public static extern bool PostMessage(IntPtr hWnd, uint msg, IntPtr wParam, IntPtr lParam);
}
'@

if ($ViewerId -eq 0) {
    $viewers = @(Get-Process -Name RadeonGPUProfiler -ErrorAction SilentlyContinue)
    if ($viewers.Count -ne 1) {
        throw "Expected one RadeonGPUProfiler process; found $($viewers.Count). Pass -ViewerId explicitly."
    }
    $ViewerId = $viewers[0].Id
}
$window = Get-RgpWindowHandle -ViewerId $ViewerId
$root = [System.Windows.Automation.AutomationElement]::FromHandle($window)
$sourceRows = @(Import-Csv -LiteralPath $EventTimingCsv)
if ($sourceRows.Count -lt 1) { throw 'Event timing CSV is empty.' }

function Get-Descendants {
    $root.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition)
}
Enter-RgpEventsTab -Root $root
function Send-EventKey([int]$VirtualKey) {
    [RgpShaderWindow]::PostMessage($window, 0x100, [IntPtr]$VirtualKey, [IntPtr]0) | Out-Null
    [RgpShaderWindow]::PostMessage($window, 0x101, [IntPtr]$VirtualKey, [IntPtr]0) | Out-Null
}
function Get-EventTab([string]$Name) {
    $element = @(Get-Descendants | Where-Object {
        $_.Current.Name -eq $Name -and
        $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::TabItem
    }) | Select-Object -First 1
    if (-not $element) { throw "RGP tab '$Name' is unavailable." }
    $pattern = [System.Windows.Automation.InvokePattern]$element.GetCurrentPattern(
        [System.Windows.Automation.InvokePattern]::Pattern)
    $pattern.Invoke()
    Start-Sleep -Milliseconds 140
}
function Get-SelectedEventId {
    $element = @(Get-Descendants | Where-Object {
        $_.Current.AutomationId -match '(BarrierSidePanel|SingleEventSidePanel)\.event_name_$'
    }) | Select-Object -First 1
    if (-not $element) { throw 'RGP selected event title is unavailable.' }
    $match = [regex]::Match($element.Current.Name, '^(\d+)\s+')
    if (-not $match.Success) { throw "Cannot parse selected RGP event '$($element.Current.Name)'." }
    [int]$match.Groups[1].Value
}
function Get-ValueByIdSuffix($Elements, [string]$Suffix) {
    $element = @($Elements | Where-Object { $_.Current.AutomationId -like ('*' + $Suffix) }) |
        Select-Object -First 1
    if ($element) { return $element.Current.Name }
    return ''
}

# One representative dispatch per named user kernel. The RGP panel also gives
# per-dispatch thread/wavefront counts, so this samples signatures, not rates.
$targets = @($sourceRows | Where-Object { $_.Event -like 'kernel_*' } |
    Group-Object { $_.Event -replace '\(.*', '' } |
    ForEach-Object {
        $first = $_.Group | Sort-Object { [int]$_.Id } | Select-Object -First 1
        [pscustomobject]@{ Id = [int]$first.Id; Event = $_.Name }
    } | Sort-Object Id)
if ($targets.Count -lt 1) { throw 'No user kernels found in the timing CSV.' }

Get-EventTab 'Event timing'
Send-EventKey 0x24
Start-Sleep -Milliseconds 100
$current = Get-SelectedEventId
$records = [System.Collections.Generic.List[object]]::new()
for ($index = 0; $index -lt $targets.Count; $index++) {
    $target = $targets[$index]
    $maxAttempts = [math]::Abs($target.Id - $current) + 8
    for ($attempt = 0; $attempt -lt $maxAttempts -and $current -ne $target.Id; $attempt++) {
        Send-EventKey $(if ($current -lt $target.Id) { 0x28 } else { 0x26 })
        Start-Sleep -Milliseconds 45
        $current = Get-SelectedEventId
    }
    if ($current -ne $target.Id) { throw "Could not select event $($target.Id); current event is $current." }

    Get-EventTab 'Wavefront occupancy'
    $elements = Get-Descendants
    $wavefronts = Get-ValueByIdSuffix $elements 'total_wavefronts_'
    $threads = Get-ValueByIdSuffix $elements 'total_threads_'
    $averageThreads = Get-ValueByIdSuffix $elements 'average_threads_'
    $table = @($elements | Where-Object { $_.Current.AutomationId -like '*usage_table_view_' }) |
        Select-Object -First 1
    if (-not $table) { throw "No shader usage table for event $($target.Id) ($($target.Event))." }
    $grid = [System.Windows.Automation.GridPattern]$table.GetCurrentPattern(
        [System.Windows.Automation.GridPattern]::Pattern)
    if ($grid.Current.ColumnCount -ne 5) { throw "Unexpected shader usage table column count for event $($target.Id)." }
    for ($row = 0; $row -lt $grid.Current.RowCount; $row++) {
        $records.Add([pscustomobject]@{
            EventId = $target.Id
            Kernel = $target.Event
            ApiShader = $grid.GetItem($row, 0).Current.Name
            TotalWavefronts = $wavefronts
            TotalThreads = $threads
            AverageThreadsPerWavefront = $averageThreads
            VGPR = $grid.GetItem($row, 1).Current.Name
            SGPR = $grid.GetItem($row, 2).Current.Name
            LDSBytes = $grid.GetItem($row, 3).Current.Name
            TheoreticalOccupancy = $grid.GetItem($row, 4).Current.Name
        })
    }
    Get-EventTab 'Event timing'
    $current = Get-SelectedEventId
    Write-Progress -Activity 'Reading RGP shader resources' -Status "$($index + 1) / $($targets.Count)" -PercentComplete (100.0 * ($index + 1) / $targets.Count)
}
Write-Progress -Activity 'Reading RGP shader resources' -Completed

$fullCsv = [IO.Path]::GetFullPath($OutCsv)
$outDir = [IO.Path]::GetDirectoryName($fullCsv)
if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
$records | Export-Csv -NoTypeInformation -Encoding utf8 -Path $fullCsv
"kernel_symbols=$($targets.Count) shader_rows=$($records.Count) csv=$fullCsv"
$records | Format-Table -AutoSize
