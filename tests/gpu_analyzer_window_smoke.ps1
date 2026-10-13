# GPT6-Astra: isolated UI lifecycle smoke test. Never controls Trainer.
param([Parameter(Mandatory=$true)][string]$Exe)
$ErrorActionPreference='Stop'
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class AnalyzerWindowTest {
 [DllImport("user32.dll",CharSet=CharSet.Unicode)] public static extern IntPtr FindWindow(string cls,string name);
 [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr w,uint m,UIntPtr a,IntPtr b);
 [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr w);
}
'@
if([AnalyzerWindowTest]::FindWindow('DQNGpuAnalyzerWindow','DQN GPU Analyzer') -ne [IntPtr]::Zero){throw 'Existing Analyzer: test refused'}
$Exe=(Resolve-Path -LiteralPath $Exe).Path
$dir=Split-Path $Exe
$p=Start-Process -FilePath $Exe -ArgumentList '--hidden --duration 20 --mode trace' -WorkingDirectory $dir -WindowStyle Hidden -PassThru
try {
 $window=[IntPtr]::Zero
 for($i=0;$i -lt 50 -and $window -eq [IntPtr]::Zero;$i++){Start-Sleep -Milliseconds 100;$window=[AnalyzerWindowTest]::FindWindow('DQNGpuAnalyzerWindow','DQN GPU Analyzer')}
 if($window -eq [IntPtr]::Zero){throw 'Missing Win32 window'}
 if([AnalyzerWindowTest]::IsWindowVisible($window)){throw 'Hidden launch failed'}
 [AnalyzerWindowTest]::PostMessage($window,0x8029,[UIntPtr]::new([uint64]1),[IntPtr]::Zero)|Out-Null
 Start-Sleep -Milliseconds 300
 if(![AnalyzerWindowTest]::IsWindowVisible($window)){throw 'Show failed'}
 [AnalyzerWindowTest]::PostMessage($window,0x10,[UIntPtr]::Zero,[IntPtr]::Zero)|Out-Null
 Start-Sleep -Milliseconds 300
 $p.Refresh()
 if($p.HasExited -or [AnalyzerWindowTest]::IsWindowVisible($window)){throw 'Close did not hide independently'}
 $second=Start-Process -FilePath $Exe -ArgumentList '--hidden --duration 1' -WorkingDirectory $dir -WindowStyle Hidden -PassThru
 if(!$second.WaitForExit(5000) -or $second.ExitCode -ne 2){throw 'Duplicate instance not rejected'}
 Start-Sleep -Seconds 2
 [AnalyzerWindowTest]::PostMessage($window,0x8029,[UIntPtr]::Zero,[IntPtr]::Zero)|Out-Null
 if(!$p.WaitForExit(15000) -or $p.ExitCode -ne 0){throw 'Graceful stop failed'}
 $log=Join-Path $dir 'logs/metric_gpu_telemetry_v2.bin'
 $size=(Get-Item $log).Length
 if($size -le 24 -or (($size-24)%52) -ne 0){throw 'Telemetry records incomplete'}
 $csv=Join-Path $dir 'smoke.csv'
 $report=Start-Process -FilePath $Exe -ArgumentList @('--csv',('"'+$log+'"'),'--output',('"'+$csv+'"')) -WorkingDirectory $dir -WindowStyle Hidden -PassThru
 if(!$report.WaitForExit(15000) -or $report.ExitCode -ne 0 -or !(Test-Path $csv)){throw 'CSV compatibility failed'}
 'PASS: hidden launch, show, close-to-hide, duplicate rejection, stop/join, complete records, CSV export'
} finally {
 $p.Refresh()
 if(!$p.HasExited){[AnalyzerWindowTest]::PostMessage($window,0x8029,[UIntPtr]::Zero,[IntPtr]::Zero)|Out-Null;$p.WaitForExit(15000)|Out-Null}
}
