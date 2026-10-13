[CmdletBinding()]
param(
    [int]$ViewerId = 0,
    [Parameter(Mandatory = $true)][string]$OutCsv,
    [int]$MaxEvents = 0
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Import-Module (Join-Path $PSScriptRoot 'rgp_uia_common.psm1') -Force
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class RgpTimingWindow {
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

function Get-Descendants {
    $root.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition)
}
Enter-RgpEventsTab -Root $root
function Send-EventKey([int]$VirtualKey) {
    [RgpTimingWindow]::PostMessage($window, 0x100, [IntPtr]$VirtualKey, [IntPtr]0) | Out-Null
    [RgpTimingWindow]::PostMessage($window, 0x101, [IntPtr]$VirtualKey, [IntPtr]0) | Out-Null
}
function Convert-RgpTimeToUs([string]$Value) {
    $match = [regex]::Match($Value, '^([0-9.,]+)\s*(ns|us|µs|μs|ms)$')
    if (-not $match.Success) { throw "Unrecognized RGP time value '$Value'." }
    $number = [double]::Parse($match.Groups[1].Value,
        [Globalization.CultureInfo]::InvariantCulture)
    switch ($match.Groups[2].Value) {
        'ns' { return $number / 1000.0 }
        'ms' { return $number * 1000.0 }
        default { return $number }
    }
}
function Get-Stats([double[]]$Values) {
    if ($Values.Count -eq 0) { throw 'Cannot summarize an empty sample.' }
    $sorted = @($Values | Sort-Object)
    $median = if (($sorted.Count % 2) -eq 0) {
        ($sorted[$sorted.Count / 2 - 1] + $sorted[$sorted.Count / 2]) / 2.0
    } else { $sorted[[int][math]::Floor($sorted.Count / 2)] }
    $p95Index = [int][math]::Ceiling(0.95 * $sorted.Count) - 1
    [pscustomobject]@{
        Count = $sorted.Count
        MeanUs = ($sorted | Measure-Object -Average).Average
        MedianUs = $median
        P95Us = $sorted[$p95Index]
        MinUs = $sorted[0]
        MaxUs = $sorted[-1]
        SumUs = ($sorted | Measure-Object -Sum).Sum
    }
}

$eventTimingTab = @(Get-Descendants | Where-Object {
    $_.Current.Name -eq 'Event timing' -and
    $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::TabItem
}) | Select-Object -First 1
if (-not $eventTimingTab) { throw 'The RGP Event timing tab is not available.' }
try {
    ([System.Windows.Automation.InvokePattern]$eventTimingTab.GetCurrentPattern(
        [System.Windows.Automation.InvokePattern]::Pattern)).Invoke()
} catch { throw 'Could not activate the RGP Event timing tab.' }
Start-Sleep -Milliseconds 200

$tree = @(Get-Descendants | Where-Object {
    $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::Tree
}) | Select-Object -First 1
if (-not $tree) { throw 'Open EVENTS > Event timing in RGP before exporting.' }
try {
    $grid = [System.Windows.Automation.GridPattern]$tree.GetCurrentPattern(
        [System.Windows.Automation.GridPattern]::Pattern)
} catch { throw 'The selected RGP event tree does not expose a readable grid.' }
$rowCount = $grid.Current.RowCount
if ($rowCount -lt 1) { throw 'RGP event tree contains no events.' }
$captureCount = if ($MaxEvents -gt 0) { [math]::Min($MaxEvents, $rowCount) } else { $rowCount }

# RGP virtualizes event-row screen bounds. Home is reliable; clicking a
# GridItem's reported bounds can select a different row.
Send-EventKey 0x24
Start-Sleep -Milliseconds 100
$records = [System.Collections.Generic.List[object]]::new()
for ($expected = 0; $expected -lt $captureCount; $expected++) {
    $elements = $null
    $nameElement = $null
    $idMatch = $null
    for ($attempt = 0; $attempt -lt 12; $attempt++) {
        $elements = Get-Descendants
        $nameElement = @($elements | Where-Object {
            $_.Current.AutomationId -match '(BarrierSidePanel|SingleEventSidePanel)\.event_name_$'
        }) | Select-Object -First 1
        if (-not $nameElement) { throw "No selected-event details at row $expected." }
        $idMatch = [regex]::Match($nameElement.Current.Name, '^(\d+)\s+(.*)$')
        if (-not $idMatch.Success) { throw "Invalid selected event '$($nameElement.Current.Name)'." }
        $actual = [int]$idMatch.Groups[1].Value
        if ($actual -eq $expected) { break }
        if ($actual -lt 0 -or $actual -ge $rowCount) { throw "Invalid event ID $actual." }
        Send-EventKey $(if ($actual -lt $expected) { 0x28 } else { 0x26 })
        Start-Sleep -Milliseconds 60
    }
    if (-not $idMatch.Success -or [int]$idMatch.Groups[1].Value -ne $expected) {
        throw "Could not stabilize RGP event $expected; current selection is '$($nameElement.Current.Name)'."
    }

    $prefix = $nameElement.Current.AutomationId -replace '\.event_name_$', ''
    $startElement = @($elements | Where-Object { $_.Current.AutomationId -eq ($prefix + '.start_time_') }) | Select-Object -First 1
    $endElement = @($elements | Where-Object { $_.Current.AutomationId -eq ($prefix + '.end_time_') }) | Select-Object -First 1
    $durationElement = @($elements | Where-Object { $_.Current.AutomationId -eq ($prefix + '.duration_') }) | Select-Object -First 1
    if (-not $startElement -or -not $endElement -or -not $durationElement) {
        throw "Missing timing details for RGP event $expected."
    }
    $records.Add([pscustomobject]@{
        Id = $expected
        Event = $idMatch.Groups[2].Value
        StartUs = Convert-RgpTimeToUs $startElement.Current.Name
        EndUs = Convert-RgpTimeToUs $endElement.Current.Name
        DurationUs = Convert-RgpTimeToUs $durationElement.Current.Name
    })
    if (($expected % 32) -eq 0 -or $expected -eq $captureCount - 1) {
        Write-Progress -Activity 'Reading RGP event timing' -Status "$($expected + 1) / $captureCount" -PercentComplete ((100.0 * ($expected + 1)) / $captureCount)
    }
}
Write-Progress -Activity 'Reading RGP event timing' -Completed

$fullCsv = [IO.Path]::GetFullPath($OutCsv)
$outDir = [IO.Path]::GetDirectoryName($fullCsv)
if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
$records | Export-Csv -NoTypeInformation -Encoding utf8 -Path $fullCsv

$summary = [System.Collections.Generic.List[object]]::new()
$records | Group-Object { if ($_.Event -eq 'CmdBarrier()') { 'CmdBarrier timeline span' } else { $_.Event -replace '\(.*', '' } } | ForEach-Object {
    $values = [double[]]@($_.Group | ForEach-Object { [double]$_.DurationUs })
    $stats = Get-Stats $values
    $summary.Add([pscustomobject]@{
        Metric = $_.Name; Count = $stats.Count
        MeanUs = [math]::Round($stats.MeanUs, 3)
        MedianUs = [math]::Round($stats.MedianUs, 3)
        P95Us = [math]::Round($stats.P95Us, 3)
        MinUs = [math]::Round($stats.MinUs, 3)
        MaxUs = [math]::Round($stats.MaxUs, 3)
        SumUs = [math]::Round($stats.SumUs, 3)
        NegativeCount = $null
    })
}

$dispatches = @($records | Where-Object { $_.Event -ne 'CmdBarrier()' })
$gaps = [System.Collections.Generic.List[double]]::new()
for ($i = 1; $i -lt $dispatches.Count; $i++) {
    $gaps.Add([double]$dispatches[$i].StartUs - [double]$dispatches[$i - 1].EndUs)
}
if ($gaps.Count -gt 0) {
    $gapStats = Get-Stats ([double[]]$gaps.ToArray())
    $positive = [double[]]@($gaps | Where-Object { $_ -ge 0 })
    $summary.Add([pscustomobject]@{
        Metric = 'Consecutive non-barrier dispatch gap'; Count = $gapStats.Count
        MeanUs = [math]::Round($gapStats.MeanUs, 3)
        MedianUs = [math]::Round($gapStats.MedianUs, 3)
        P95Us = [math]::Round($gapStats.P95Us, 3)
        MinUs = [math]::Round($gapStats.MinUs, 3)
        MaxUs = [math]::Round($gapStats.MaxUs, 3)
        SumUs = [math]::Round($gapStats.SumUs, 3)
        NegativeCount = @($gaps | Where-Object { $_ -lt 0 }).Count
    })
    $summary.Add([pscustomobject]@{
        Metric = 'Positive gaps > 100 us'; Count = @($positive | Where-Object { $_ -gt 100 }).Count
        MeanUs = $null; MedianUs = $null; P95Us = $null; MinUs = $null; MaxUs = $null; SumUs = $null
        NegativeCount = $null
    })
    $summary.Add([pscustomobject]@{
        Metric = 'Positive gaps > 1000 us'; Count = @($positive | Where-Object { $_ -gt 1000 }).Count
        MeanUs = $null; MedianUs = $null; P95Us = $null; MinUs = $null; MaxUs = $null; SumUs = $null
        NegativeCount = $null
    })
}

$summaryPath = [IO.Path]::ChangeExtension($fullCsv, '.summary.csv')
$summary | Export-Csv -NoTypeInformation -Encoding utf8 -Path $summaryPath
"events=$($records.Count) / $rowCount dispatches=$($dispatches.Count) timing_csv=$fullCsv summary_csv=$summaryPath"
$summary | Where-Object { $_.Metric -match 'kernel_(attn_backward|pool_backward|fc_delta|fc)$|dispatch gap|Positive gaps' } | Format-Table -AutoSize
