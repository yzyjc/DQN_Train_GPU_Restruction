if (-not ('RgpUiaWindowFinder' -as [type])) {
    Add-Type @'
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class RgpUiaWindowFinder {
    public delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);
    [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);
    [DllImport("user32.dll", SetLastError=true)] public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId);
    [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr hwnd, StringBuilder text, int maxCount);
}
'@
}

function Get-RgpWindowHandle {
    [CmdletBinding()]
    param([Parameter(Mandatory=$true)][int]$ViewerId)

    $windowCandidates = [System.Collections.Generic.List[object]]::new()
    $callback = [RgpUiaWindowFinder+EnumWindowsProc]{
        param($hwnd, $lParam)
        [uint32]$processId = 0
        [RgpUiaWindowFinder]::GetWindowThreadProcessId($hwnd, [ref]$processId) | Out-Null
        if ($processId -eq $ViewerId) {
            $title = [System.Text.StringBuilder]::new(512)
            [RgpUiaWindowFinder]::GetWindowText($hwnd, $title, $title.Capacity) | Out-Null
            if ($title.ToString() -match 'Radeon GPU Profiler') {
                $windowCandidates.Add([pscustomobject]@{ Handle = $hwnd; Title = $title.ToString() })
            }
        }
        return $true
    }
    [RgpUiaWindowFinder]::EnumWindows($callback, [IntPtr]::Zero) | Out-Null
    if ($windowCandidates.Count -ne 1) {
        throw "Expected one RGP main window for PID $ViewerId; found $($windowCandidates.Count)."
    }
    return [IntPtr]$windowCandidates[0].Handle
}

function Get-RgpAutomationRoot {
    [CmdletBinding()]
    param([Parameter(Mandatory=$true)][int]$ViewerId)

    $handle = Get-RgpWindowHandle -ViewerId $ViewerId
    return [System.Windows.Automation.AutomationElement]::FromHandle($handle)
}

function Enter-RgpEventsTab {
    [CmdletBinding()]
    param([Parameter(Mandatory=$true)][System.Windows.Automation.AutomationElement]$Root)

    $condition = [System.Windows.Automation.Condition]::TrueCondition
    $tabs = @($Root.FindAll([System.Windows.Automation.TreeScope]::Descendants, $condition) |
        Where-Object {
            $_.Current.Name -eq 'EVENTS' -and
            $_.Current.ControlType -eq [System.Windows.Automation.ControlType]::TabItem
        } | Sort-Object { $_.Current.BoundingRectangle.Y })
    if ($tabs.Count -eq 0) { throw 'The RGP EVENTS tab is unavailable.' }
    try {
        ([System.Windows.Automation.InvokePattern]$tabs[0].GetCurrentPattern(
            [System.Windows.Automation.InvokePattern]::Pattern)).Invoke()
    } catch { throw 'Could not activate the RGP EVENTS tab.' }
    Start-Sleep -Milliseconds 200
}

Export-ModuleMember -Function Get-RgpWindowHandle, Get-RgpAutomationRoot, Enter-RgpEventsTab
