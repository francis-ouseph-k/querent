# mcp_servers/stop_mcp_servers.ps1
# Stop all NL→SQL MCP server background jobs.

Write-Host "Stopping MCP servers..." -ForegroundColor Yellow

$jobs = Get-Job | Where-Object { $_.Name -like "mcp_*" }
if ($jobs.Count -eq 0) {
    Write-Host "  No MCP server jobs found." -ForegroundColor Yellow
} else {
    foreach ($job in $jobs) {
        Stop-Job  -Job $job
        Remove-Job -Job $job
        Write-Host "  [STOPPED] $($job.Name)" -ForegroundColor Green
    }
}
Write-Host "Done." -ForegroundColor Cyan
