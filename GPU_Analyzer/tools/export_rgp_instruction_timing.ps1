[CmdletBinding()]
param(
    [int]$ViewerId = 0,
    [Parameter(Mandatory=$true)][string]$EventTimingCsv,
    [Parameter(Mandatory=$true)][string]$OutCsv,
    [ValidateRange(1,256)][int]$MaxKernels = 0,
    [ValidateRange(1,32)][int]$MaxEventsPerKernel = 4
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName UIAutomationClient
Add-Type -AssemblyName UIAutomationTypes
Import-Module (Join-Path $PSScriptRoot 'rgp_uia_common.psm1') -Force
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class RgpInstructionWindow {
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
$eventRows = @(Import-Csv -LiteralPath $EventTimingCsv)
if ($eventRows.Count -lt 1) { throw 'Event timing CSV is empty.' }

function Get-Descendants {
    $root.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition)
}
function Send-EventKey([int]$VirtualKey) {
    [RgpInstructionWindow]::PostMessage($window, 0x100, [IntPtr]$VirtualKey, [IntPtr]0) | Out-Null
    [RgpInstructionWindow]::PostMessage($window, 0x101, [IntPtr]$VirtualKey, [IntPtr]0) | Out-Null
    Start-Sleep -Milliseconds 30
}
function Select-Tab([string]$Name) {
    $tab = @(Get-Descendants | Where-Object {
        $_.Current.Name -eq $Name -and
        $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::TabItem
    } | Sort-Object { $_.Current.BoundingRectangle.Y } | Select-Object -Last 1) | Select-Object -First 1
    if (-not $tab) { throw "RGP tab '$Name' is unavailable." }
    ([System.Windows.Automation.InvokePattern]$tab.GetCurrentPattern(
        [System.Windows.Automation.InvokePattern]::Pattern)).Invoke()
    Start-Sleep -Milliseconds 150
}
function Get-ElementNameBySuffix($Elements, [string]$Suffix) {
    $element = @($Elements | Where-Object { $_.Current.AutomationId -like ('*' + $Suffix) }) |
        Select-Object -First 1
    if ($element) { return $element.Current.Name }
    return ''
}
function Get-SelectedEventName($Elements) {
    $element = @($Elements | Where-Object {
        $_.Current.AutomationId -match '(BarrierSidePanel|SingleEventSidePanel)\.event_name_$'
    }) | Select-Object -First 1
    if (-not $element) { return '' }
    return $element.Current.Name
}
function Get-SelectedEventId {
    $name = Get-SelectedEventName (Get-Descendants)
    $match = [regex]::Match($name, '^(\d+)\s+')
    if (-not $match.Success) { throw "Cannot parse selected RGP event '$name'." }
    return [int]$match.Groups[1].Value
}
function Select-EventId([int]$TargetId, [int]$CurrentId) {
    $maxAttempts = [math]::Abs($TargetId - $CurrentId) + 12
    for ($attempt = 0; $attempt -lt $maxAttempts -and $CurrentId -ne $TargetId; $attempt++) {
        [RgpInstructionWindow]::PostMessage($window, 0x100,
            [IntPtr]$(if ($CurrentId -lt $TargetId) { 0x28 } else { 0x26 }), [IntPtr]0) | Out-Null
        [RgpInstructionWindow]::PostMessage($window, 0x101,
            [IntPtr]$(if ($CurrentId -lt $TargetId) { 0x28 } else { 0x26 }), [IntPtr]0) | Out-Null
        Start-Sleep -Milliseconds 60
        $CurrentId = Get-SelectedEventId
    }
    if ($CurrentId -ne $TargetId) { throw "Could not select event $TargetId; current event is $CurrentId." }
    return $CurrentId
}

$kernelGroups = @($eventRows | Where-Object { $_.Event -like 'kernel_*' } |
    Group-Object { $_.Event -replace '\(.*', '' } | Sort-Object Name)
if ($MaxKernels -gt 0) { $kernelGroups = @($kernelGroups | Select-Object -First $MaxKernels) }
$targets = @($kernelGroups | ForEach-Object {
    $kernelName = $_.Name
    $_.Group | Sort-Object { [int]$_.Id } | Select-Object -First $MaxEventsPerKernel |
        ForEach-Object { [pscustomobject]@{ Id = [int]$_.Id; Event = $kernelName } }
} | Sort-Object Id)
if ($targets.Count -lt 1) { throw 'No user kernels found in the event timing CSV.' }

Enter-RgpEventsTab -Root $root
Select-Tab 'Event timing'
Send-EventKey 0x24
$currentId = Get-SelectedEventId
$currentId = Select-EventId -TargetId 0 -CurrentId $currentId
Select-Tab 'Instruction timing'
$instructionRecords = [System.Collections.Generic.List[object]]::new()
$summaryRecords = [System.Collections.Generic.List[object]]::new()

for ($targetIndex = 0; $targetIndex -lt $targets.Count; $targetIndex++) {
    $target = $targets[$targetIndex]
    Select-Tab 'Event timing'
    $currentId = Get-SelectedEventId
    $currentId = Select-EventId -TargetId $target.Id -CurrentId $currentId

    Select-Tab 'Instruction timing'
    $elements = Get-Descendants
    $eventTitle = Get-ElementNameBySuffix $elements 'event_text_label_'
    if (-not $eventTitle) {
        $invalidTitle = Get-ElementNameBySuffix $elements 'invalid_event_page_title_'
        if ($invalidTitle -match ('^Event\s+' + $target.Id + '\s+No instruction timing data')) {
            $summaryRecords.Add([pscustomobject]@{
                EventId = $target.Id; Kernel = $target.Event; InstructionData = $false
                Reason = 'RGP has no instruction trace for this event/shader engine'
                Shader = ''; ShaderDuration = ''; AnalyzedWavefronts = ''; TotalWavefronts = ''
                TheoreticalOccupancy = ''; VGPR = ''; SGPR = ''; LDS = ''
                HardwareUtilizationNumeric = $false; InstructionRows = 0
            })
            $currentId = $target.Id
            continue
        }
    }
    $titleMatch = [regex]::Match($eventTitle, '^(\d+)\s+')
    if (-not $titleMatch.Success -or [int]$titleMatch.Groups[1].Value -ne $target.Id) {
        throw "Instruction pane is showing '$eventTitle', expected event $($target.Id)."
    }
    $isaTree = @($elements | Where-Object {
        $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::Tree -and
        $_.Current.AutomationId -match 'isa_tree_view_'
    }) | Select-Object -First 1
    if (-not $isaTree) {
        $noData = [bool](@($elements | Where-Object { $_.Current.Name -match 'No instruction timing data' }).Count -gt 0)
        $summaryRecords.Add([pscustomobject]@{
            EventId = $target.Id; Kernel = $target.Event; InstructionData = $false
            Reason = $(if ($noData) { 'RGP has no instruction trace for this event/shader engine' } else { 'RGP ISA table unavailable' })
            Shader = ''; ShaderDuration = ''; AnalyzedWavefronts = ''; TotalWavefronts = ''
            TheoreticalOccupancy = ''; VGPR = ''; SGPR = ''; LDS = ''
            HardwareUtilizationNumeric = $false; InstructionRows = 0
        })
        $currentId = $target.Id
        continue
    }

    $grid = [System.Windows.Automation.GridPattern]$isaTree.GetCurrentPattern(
        [System.Windows.Automation.GridPattern]::Pattern)
    if ($grid.Current.ColumnCount -ne 6 -or $grid.Current.RowCount -lt 2) {
        throw "Unexpected RGP ISA grid for event $($target.Id): $($grid.Current.RowCount)x$($grid.Current.ColumnCount)."
    }
    $shader = $grid.GetItem(0, 1).Current.Name
    $rowsWritten = 0
    for ($row = 1; $row -lt $grid.Current.RowCount; $row++) {
        $cells = @(); for ($column = 0; $column -lt 6; $column++) {
            $cells += $grid.GetItem($row, $column).Current.Name
        }
        $instructionRecords.Add([pscustomobject]@{
            EventId = $target.Id; Kernel = $target.Event; Shader = $shader
            InstructionIndex = $cells[0]; Opcode = $cells[1]; Operands = $cells[2]
            HitCount = $cells[3]; InstructionCostPercent = $cells[4]; LatencyClocks = $cells[5]
        })
        $rowsWritten++
    }
    $elements = Get-Descendants
    $wavefronts = Get-ElementNameBySuffix $elements 'wavefronts_value_'
    $waveMatch = [regex]::Match($wavefronts, '^(\d+)\s+out of\s+(\d+)')
    $summaryRecords.Add([pscustomobject]@{
        EventId = $target.Id; Kernel = $target.Event; InstructionData = $true; Reason = ''
        Shader = $shader; ShaderDuration = (Get-ElementNameBySuffix $elements 'shader_duration_value_')
        AnalyzedWavefronts = $(if ($waveMatch.Success) { $waveMatch.Groups[1].Value } else { '' })
        TotalWavefronts = $(if ($waveMatch.Success) { $waveMatch.Groups[2].Value } else { '' })
        TheoreticalOccupancy = (Get-ElementNameBySuffix $elements 'theoretical_occupancy_value_')
        VGPR = (Get-ElementNameBySuffix $elements 'vector_register_value_')
        SGPR = (Get-ElementNameBySuffix $elements 'scalar_register_value_')
        LDS = (Get-ElementNameBySuffix $elements 'local_data_share_value_')
        HardwareUtilizationNumeric = $false; InstructionRows = $rowsWritten
    })
    $currentId = $target.Id
    Write-Progress -Activity 'Reading RGP instruction timing' -Status "$($targetIndex + 1) / $($targets.Count)" -PercentComplete (100.0 * ($targetIndex + 1) / $targets.Count)
}
Write-Progress -Activity 'Reading RGP instruction timing' -Completed

$fullCsv = [IO.Path]::GetFullPath($OutCsv)
$outDir = [IO.Path]::GetDirectoryName($fullCsv)
if (-not (Test-Path -LiteralPath $outDir)) { New-Item -ItemType Directory -Path $outDir -Force | Out-Null }
$instructionRecords | Export-Csv -NoTypeInformation -Encoding utf8 -Path $fullCsv
$summaryPath = [IO.Path]::ChangeExtension($fullCsv, '.summary.csv')
$summaryRecords | Export-Csv -NoTypeInformation -Encoding utf8 -Path $summaryPath
"events_checked=$($targets.Count) shaders_with_trace=$(@($summaryRecords | Where-Object InstructionData).Count) instruction_rows=$($instructionRecords.Count) csv=$fullCsv summary=$summaryPath"
$summaryRecords | Format-Table EventId, Kernel, InstructionData, ShaderDuration, AnalyzedWavefronts, TotalWavefronts, InstructionRows -AutoSize
