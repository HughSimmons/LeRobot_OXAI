# Holds a Windows "system required" power request (no idle sleep) until killed. Lid-close is separate (powercfg).
$t = Add-Type -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);' -Name K -Namespace W -PassThru
while ($true) { $r = $t::SetThreadExecutionState([uint32]2147483649); Write-Output ("keep-awake ret=" + $r); Start-Sleep 30 }
