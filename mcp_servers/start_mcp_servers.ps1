# mcp_servers/start_mcp_servers.ps1
# ─────────────────────────────────────
# Start all four NL→SQL MCP servers as background jobs.
# Run from the project root with the venv active:
#
#   venv1\Scripts\Activate.ps1
#   .\mcp_servers\start_mcp_servers.ps1
#
# To stop all servers:
#   .\mcp_servers\stop_mcp_servers.ps1
#
# Ports:
#   5010  qdrant_server      dense vector search
#   5011  opensearch_server  BM25 keyword search
#   5012  postgres_server    read-only SQL execution
#   5013  corpus_server      failure corpus management

$ErrorActionPreference = "Stop"

Write-Host "Starting NL→SQL MCP servers..." -ForegroundColor Cyan

# Qdrant server — port 5010
$qdrant = Start-Job -Name "mcp_qdrant" -ScriptBlock {
    Set-Location $using:PWD
    & python mcp_servers\qdrant_server.py
}
Write-Host "  [OK] qdrant_server     → http://127.0.0.1:5010  (job $($qdrant.Id))"

# OpenSearch server — port 5011
$opensearch = Start-Job -Name "mcp_opensearch" -ScriptBlock {
    Set-Location $using:PWD
    & python mcp_servers\opensearch_server.py
}
Write-Host "  [OK] opensearch_server → http://127.0.0.1:5011  (job $($opensearch.Id))"

# PostgreSQL server — port 5012
$postgres = Start-Job -Name "mcp_postgres" -ScriptBlock {
    Set-Location $using:PWD
    & python mcp_servers\postgres_server.py
}
Write-Host "  [OK] postgres_server   → http://127.0.0.1:5012  (job $($postgres.Id))"

# Corpus server — port 5013
$corpus = Start-Job -Name "mcp_corpus" -ScriptBlock {
    Set-Location $using:PWD
    & python mcp_servers\corpus_server.py
}
Write-Host "  [OK] corpus_server     → http://127.0.0.1:5013  (job $($corpus.Id))"

Write-Host ""
Write-Host "All MCP servers started. Waiting 3s for startup..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Quick health check
Write-Host ""
Write-Host "Health checks:" -ForegroundColor Cyan
$checks = @(
    @{ name = "qdrant";      url = "http://127.0.0.1:5010/health" },
    @{ name = "opensearch";  url = "http://127.0.0.1:5011/health" },
    @{ name = "postgres";    url = "http://127.0.0.1:5012/health" },
    @{ name = "corpus";      url = "http://127.0.0.1:5013/health" }
)
foreach ($c in $checks) {
    try {
        $r = Invoke-WebRequest -Uri $c.url -TimeoutSec 2 -UseBasicParsing
        Write-Host "  [OK] $($c.name)" -ForegroundColor Green
    } catch {
        Write-Host "  [--] $($c.name) not responding yet (may still be starting)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Run 'python main.py' to start the NL→SQL application." -ForegroundColor Cyan
Write-Host "Run '.\mcp_servers\stop_mcp_servers.ps1' to stop all servers." -ForegroundColor Cyan
