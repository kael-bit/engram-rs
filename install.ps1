# engram installer for Windows
# Usage: irm https://raw.githubusercontent.com/kael-bit/engram-rs/main/install.ps1 | iex

$ErrorActionPreference = 'Stop'
$REPO = 'kael-bit/engram-rs'
$DEFAULT_PORT = 3917
$DEFAULT_DB = [IO.Path]::Combine($env:USERPROFILE, '.engram', 'engram.db')

function Info($msg)  { Write-Host '  -> ' -ForegroundColor Blue -NoNewline; Write-Host $msg }
function Ok($msg)    { Write-Host '  OK ' -ForegroundColor Green -NoNewline; Write-Host $msg }
function Warn($msg)  { Write-Host '  !! ' -ForegroundColor Yellow -NoNewline; Write-Host $msg }
function Err($msg)   { Write-Host '  XX ' -ForegroundColor Red -NoNewline; Write-Host $msg }
function Ask($msg)   { Write-Host $msg -NoNewline -ForegroundColor White }

function Fetch-Version {
    try {
        $resp = Invoke-RestMethod ('https://api.github.com/repos/' + $REPO + '/tags')
        $latest = ($resp | Select-Object -First 1).name -replace '^v', ''
        if (-not $latest) { throw 'empty' }
        return $latest
    } catch {
        Err 'Failed to fetch latest version from GitHub'
        exit 1
    }
}

function Has($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Add-ToPath($dir) {
    $currentPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($currentPath -notlike ('*' + $dir + '*')) {
        [Environment]::SetEnvironmentVariable('Path', ($dir + ';' + $currentPath), 'User')
        $env:Path = $dir + ';' + $env:Path
        Info ('Added ' + $dir + ' to user PATH')
    }
}

function Install-Binary {
    $arch = 'x86_64'
    if (-not [Environment]::Is64BitOperatingSystem) { Err '32-bit not supported'; exit 1 }

    $binaryName = 'engram-windows-' + $arch + '.exe'
    $url = 'https://github.com/' + $REPO + '/releases/download/v' + $script:VERSION + '/' + $binaryName
    $installDir = [IO.Path]::Combine($env:USERPROFILE, '.engram', 'bin')
    $target = [IO.Path]::Combine($installDir, 'engram.exe')

    if (-not (Test-Path $installDir)) { New-Item -ItemType Directory -Path $installDir -Force | Out-Null }

    if (Test-Path $target) {
        try { $existingVer = & $target --version 2>$null } catch { $existingVer = 'unknown' }
        Warn ('engram already exists at ' + $target + ' (' + $existingVer + ')')
        Ask 'Re-download? [y/N]: '
        $r = Read-Host
        if ($r -ne 'y' -and $r -ne 'Y') {
            Ok 'Keeping existing binary'
            Add-ToPath $installDir
            return
        }
    }

    Info ('Downloading engram v' + $script:VERSION + '...')
    try {
        Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing
    } catch {
        Err ('Download failed: ' + $_)
        exit 1
    }

    Add-ToPath $installDir
    Ok ('Installed to ' + $target)
}

function Install-Cargo {
    Info 'Building from source (this may take a few minutes)...'
    cargo install --git ('https://github.com/' + $REPO + '.git') --tag ('v' + $script:VERSION)
    Ok 'Built and installed via cargo'
}

function Choose-Method {
    Write-Host ''
    Write-Host '  How would you like to install engram?' -ForegroundColor White
    Write-Host ''
    Write-Host '    1) Download pre-built binary (fastest)'

    $options = @('binary')
    $n = 2
    if (Has 'cargo') {
        Write-Host ('    ' + $n + ') Build from source with cargo')
        $options += 'cargo'
        $n++
    }

    Write-Host ''
    Ask '  Choose [1]: '
    $choice = Read-Host
    if (-not $choice) { $choice = '1' }
    $idx = [int]$choice - 1
    if ($idx -lt 0 -or $idx -ge $options.Count) { $idx = 0 }
    return $options[$idx]
}

function Configure {
    $envFile = [IO.Path]::Combine($env:USERPROFILE, '.engram', 'env.ps1')

    if (Test-Path $envFile) {
        Write-Host ''
        Info 'Existing configuration found:'
        Write-Host ''
        Get-Content $envFile | Where-Object { $_ -and $_ -notmatch '^\s*#' } | ForEach-Object {
            $line = $_
            if ($line -match '^\$env:(\w+)\s*=\s*"(.*)"') {
                $k = $Matches[1]
                $v = $Matches[2]
                if ($k -like '*KEY*' -and $v.Length -gt 8) {
                    $v = $v.Substring(0,4) + '...' + $v.Substring($v.Length - 4)
                }
                Write-Host ('      ' + $k + ' = ' + $v)
            }
        }
        Write-Host ''
        Ask '  Use existing configuration? [Y/n]: '
        $use = Read-Host
        if ($use -ne 'n' -and $use -ne 'N') {
            . $envFile
            $script:PORT = if ($env:ENGRAM_PORT) { $env:ENGRAM_PORT } else { $DEFAULT_PORT }
            $script:DB = if ($env:ENGRAM_DB) { $env:ENGRAM_DB } else { $DEFAULT_DB }
            $script:LLM_URL = $env:ENGRAM_LLM_URL
            $script:LLM_KEY = $env:ENGRAM_LLM_KEY
            $script:LLM_MODEL = $env:ENGRAM_LLM_MODEL
            $script:EMBED_URL = $env:ENGRAM_EMBED_URL
            $script:EMBED_KEY = $env:ENGRAM_EMBED_KEY
            $script:EMBED_MODEL = $env:ENGRAM_EMBED_MODEL
            $script:SKIP_ENV = $true
            Ok 'Using existing config'
            return
        }
    }

    $script:SKIP_ENV = $false
    Write-Host ''
    Write-Host '  Configuration' -ForegroundColor White
    Write-Host ''

    Ask ('  Port [' + $DEFAULT_PORT + ']: ')
    $p = Read-Host
    $script:PORT = if ($p) { $p } else { $DEFAULT_PORT }

    Ask ('  Database path [' + $DEFAULT_DB + ']: ')
    $d = Read-Host
    $script:DB = if ($d) { $d } else { $DEFAULT_DB }

    $dbDir = Split-Path $script:DB -Parent
    if (-not (Test-Path $dbDir)) { New-Item -ItemType Directory -Path $dbDir -Force | Out-Null }

    if (Test-Path $script:DB) {
        $sz = '{0:N1} MB' -f ((Get-Item $script:DB).Length / 1MB)
        Warn ('Database exists at ' + $script:DB + ' (' + $sz + ')')
        Ask '  Use existing? [Y/n]: '
        $u = Read-Host
        if ($u -eq 'n' -or $u -eq 'N') {
            Ask '  New path: '
            $np = Read-Host
            if ($np) { $script:DB = $np }
        }
    }

    Write-Host ''
    Write-Host '  LLM Configuration (optional)' -ForegroundColor White
    Write-Host ''
    Write-Host '    engram works without an LLM (embedding-only mode).'
    Write-Host '    With an LLM, it auto-triages, merges, and audits memories.'
    Write-Host ''

    Ask '  LLM API URL (or leave empty to skip): '
    $script:LLM_URL = Read-Host

    if ($script:LLM_URL) {
        Ask '  API Key: '
        $script:LLM_KEY = Read-Host -MaskInput
        Ask '  Model [gpt-4o-mini]: '
        $m = Read-Host
        $script:LLM_MODEL = if ($m) { $m } else { 'gpt-4o-mini' }
    } else {
        $script:LLM_KEY = ''
        $script:LLM_MODEL = ''
    }

    Write-Host ''
    if ($script:LLM_URL) {
        Ask '  Embedding API URL [same as LLM]: '
        $eu = Read-Host
        $script:EMBED_URL = if ($eu) { $eu } else { $script:LLM_URL }
        $script:EMBED_KEY = $script:LLM_KEY
    } else {
        Write-Host '    engram needs an embedding API for semantic search.'
        Write-Host ''
        while ($true) {
            Ask '  Embedding API URL (required): '
            $script:EMBED_URL = Read-Host
            if ($script:EMBED_URL) { break }
            Warn 'Required.'
        }
        Ask '  Embedding API Key: '
        $script:EMBED_KEY = Read-Host -MaskInput
    }

    Ask '  Embedding model [text-embedding-3-small]: '
    $em = Read-Host
    $script:EMBED_MODEL = if ($em) { $em } else { 'text-embedding-3-small' }
}

function Generate-Env {
    if ($script:SKIP_ENV) { return }

    $dir = [IO.Path]::Combine($env:USERPROFILE, '.engram')
    $file = [IO.Path]::Combine($dir, 'env.ps1')
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }

    if (Test-Path $file) {
        Warn ('Config exists at ' + $file)
        Ask '  Overwrite? [y/N]: '
        $ow = Read-Host
        if ($ow -ne 'y' -and $ow -ne 'Y') { Info 'Keeping existing'; return }
    }

    $out = @()
    $out += '# engram config'
    $out += ('$env:ENGRAM_PORT = "' + $script:PORT + '"')
    $out += ('$env:ENGRAM_DB = "' + $script:DB + '"')
    if ($script:LLM_URL) {
        $out += ('$env:ENGRAM_LLM_URL = "' + $script:LLM_URL + '"')
        $out += ('$env:ENGRAM_LLM_KEY = "' + $script:LLM_KEY + '"')
        $out += ('$env:ENGRAM_LLM_MODEL = "' + $script:LLM_MODEL + '"')
    }
    if ($script:EMBED_URL) {
        $out += ('$env:ENGRAM_EMBED_URL = "' + $script:EMBED_URL + '"')
        $out += ('$env:ENGRAM_EMBED_KEY = "' + $script:EMBED_KEY + '"')
        $out += ('$env:ENGRAM_EMBED_MODEL = "' + $script:EMBED_MODEL + '"')
    }
    ($out -join "`r`n") | Out-File -FilePath $file -Encoding utf8
    Ok ('Config saved to ' + $file)
}

function Setup-MCP {
    Write-Host ''
    Ask '  Configure MCP for a client? [y/N]: '
    $do = Read-Host
    if ($do -ne 'y' -and $do -ne 'Y') { return }

    Write-Host ''
    Write-Host '    1) Claude Code'
    Write-Host '    2) Claude Desktop'
    Write-Host '    3) Cursor'
    Write-Host '    4) Skip'
    Write-Host ''
    Ask '  Choose [1]: '
    $ch = Read-Host
    if (-not $ch) { $ch = '1' }

    $port = $script:PORT
    $json = '{ "mcpServers": { "engram": { "command": "npx", "args": ["-y", "engram-rs-mcp"], "env": { "ENGRAM_URL": "http://localhost:' + $port + '" } } } }'

    if ($ch -eq '1') {
        $dir = [IO.Path]::Combine($env:USERPROFILE, '.claude')
        $f = [IO.Path]::Combine($dir, 'claude_desktop_config.json')
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        if (Test-Path $f) { Warn ('Exists: ' + $f + ' — merge manually:'); Write-Host $json }
        else { $json | Out-File -FilePath $f -Encoding utf8; Ok ('Written to ' + $f) }
    }
    elseif ($ch -eq '2') {
        $dir = [IO.Path]::Combine($env:APPDATA, 'Claude')
        $f = [IO.Path]::Combine($dir, 'claude_desktop_config.json')
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        if (Test-Path $f) { Warn 'Config exists — merge manually:'; Write-Host $json }
        else { $json | Out-File -FilePath $f -Encoding utf8; Ok 'Claude Desktop config written' }
    }
    elseif ($ch -eq '3') {
        $dir = [IO.Path]::Combine($env:USERPROFILE, '.cursor')
        $f = [IO.Path]::Combine($dir, 'mcp.json')
        if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
        if (Test-Path $f) { Warn ('Exists: ' + $f + ' — merge manually:'); Write-Host $json }
        else { $json | Out-File -FilePath $f -Encoding utf8; Ok 'Cursor MCP config written' }
    }
}

function Start-Engram {
    Write-Host ''
    Write-Host '  Ready to start!' -ForegroundColor White
    Write-Host ''
    Ask '  Start engram now? [Y/n]: '
    $do = Read-Host
    if ($do -ne 'n' -and $do -ne 'N') {
        Info ('Starting on port ' + $script:PORT + '...')
        $envFile = [IO.Path]::Combine($env:USERPROFILE, '.engram', 'env.ps1')
        if (Test-Path $envFile) { . $envFile }
        $logFile = [IO.Path]::Combine($env:USERPROFILE, '.engram', 'engram.log')
        $errFile = [IO.Path]::Combine($env:USERPROFILE, '.engram', 'engram-err.log')
        Start-Process -FilePath ([IO.Path]::Combine($env:USERPROFILE, '.engram', 'bin', 'engram.exe')) -WindowStyle Hidden -RedirectStandardOutput $logFile -RedirectStandardError $errFile
        Start-Sleep -Seconds 1
        try {
            Invoke-RestMethod ('http://localhost:' + $script:PORT + '/health') -TimeoutSec 3 | Out-Null
            Ok ('engram running at http://localhost:' + $script:PORT)
        } catch {
            Warn 'Started but health check failed — check ~/.engram/engram.log'
        }
    }
}

function Show-Summary {
    Write-Host ''
    Write-Host '  ====================================' -ForegroundColor White
    Write-Host ('    engram v' + $script:VERSION + ' installed!') -ForegroundColor Green
    Write-Host '  ====================================' -ForegroundColor White
    Write-Host ''
    Write-Host ('    API:    http://localhost:' + $script:PORT)
    Write-Host ('    DB:     ' + $script:DB)
    Write-Host '    Config: ~/.engram/env.ps1'
    Write-Host ''
    Write-Host '    Quick test:'
    Write-Host ('      Invoke-RestMethod http://localhost:' + $script:PORT + '/health')
    Write-Host ''
    Write-Host '    Stop:  Stop-Process -Name engram'
    Write-Host '    Logs:  ~/.engram/engram.log'
    Write-Host ('    Docs:  https://github.com/' + $REPO)
    Write-Host ''
}

function Main {
    Write-Host ''
    Write-Host '    engram - memory for agents' -ForegroundColor White
    Write-Host ''

    Info 'Checking latest version...'
    $script:VERSION = Fetch-Version
    Ok ('Latest version: v' + $script:VERSION)

    $script:METHOD = Choose-Method

    if ($script:METHOD -eq 'binary') { Install-Binary }
    elseif ($script:METHOD -eq 'cargo') { Install-Cargo }

    Configure
    Generate-Env
    Setup-MCP
    Start-Engram
    Show-Summary
}

Main
