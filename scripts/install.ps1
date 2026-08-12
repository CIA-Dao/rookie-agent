[CmdletBinding()]
param(
    [Parameter()]
    [string]$Repository = 'https://github.com/CIA-Dao/rookie-agent',

    [Parameter()]
    [ValidatePattern('^v[0-9]+\.[0-9]+\.[0-9]+$')]
    [string]$Version = 'v0.0.2',

    [Parameter()]
    [switch]$SkipUvInstall,

    [Parameter()]
    [switch]$SkipPathUpdate,

    [Parameter(DontShow = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$AcceptanceCommit
)

$ErrorActionPreference = 'Stop'
$script:UvExecutable = $null

function Write-Step {
    param([string]$Message)
    Write-Host "[rookie-agent] $Message"
}

function Invoke-Uv {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

    & $script:UvExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "uv $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
    }
}

function Resolve-UvExecutable {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $candidate = Join-Path $env:USERPROFILE '.local\bin\uv.exe'
    if (Test-Path -LiteralPath $candidate) {
        return $candidate
    }

    return $null
}

function Get-UvToolState {
    param([Parameter(Mandatory = $true)][string]$Name)

    $toolRoot = (& $script:UvExecutable tool dir).Trim()
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not locate the uv tool directory.'
    }

    $environmentPath = Join-Path $toolRoot $Name
    if (-not (Test-Path -LiteralPath $environmentPath)) {
        return $null
    }

    $receiptPath = Join-Path $environmentPath 'uv-receipt.toml'
    $receipt = ''
    if (Test-Path -LiteralPath $receiptPath) {
        $receipt = Get-Content -LiteralPath $receiptPath -Raw
    }

    $listOutput = (& $script:UvExecutable tool list --show-version-specifiers --color never) -join "`n"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not inspect installed uv tools.'
    }

    $escapedName = [Regex]::Escape($Name)
    $match = [Regex]::Match(
        $listOutput,
        "(?m)^$escapedName\s+v(?<version>\S+)(?:\s+\[required:\s+(?<required>.+?)\])?\s*$"
    )

    $installedVersion = $null
    $restoreSpec = $null
    if ($match.Success) {
        $installedVersion = $match.Groups['version'].Value
        if ($match.Groups['required'].Success) {
            $restoreSpec = $match.Groups['required'].Value
        }
        else {
            $restoreSpec = "$Name==$installedVersion"
        }
    }

    [PSCustomObject]@{
        Name = $Name
        EnvironmentPath = $environmentPath
        ReceiptPath = $receiptPath
        Editable = $receipt -match '(?m)\beditable\s*='
        Version = $installedVersion
        RestoreSpec = $restoreSpec
    }
}

function Restore-UvTool {
    param([Parameter(Mandatory = $true)]$State)

    if ([string]::IsNullOrWhiteSpace($State.RestoreSpec)) {
        Write-Warning "Cannot automatically restore $($State.Name): its original install source is unknown."
        return $false
    }

    Write-Warning "Restoring $($State.Name) from its previous source."
    & $script:UvExecutable tool install --from $State.RestoreSpec $State.Name --force
    return $LASTEXITCODE -eq 0
}

function Install-RookieAgent {
    param([Parameter(Mandatory = $true)][string]$Source)

    & $script:UvExecutable tool install --from $Source rookie-agent --force
    if ($LASTEXITCODE -ne 0) {
        throw "Rookie Agent installation failed with exit code $LASTEXITCODE."
    }
}

if ($Repository -notmatch '^https://github\.com/[^/]+/[^/]+/?$') {
    throw 'Repository must be an HTTPS GitHub repository URL.'
}

$repositoryUri = [Uri]$Repository.TrimEnd('/')
$repository = $repositoryUri.AbsoluteUri.TrimEnd('/')
$sourceRef = if ([string]::IsNullOrWhiteSpace($AcceptanceCommit)) {
    $Version
}
else {
    Write-Step "Acceptance mode: installing the audited commit $AcceptanceCommit."
    $AcceptanceCommit
}
$source = "$repository/archive/$sourceRef.zip"
$expectedVersion = $Version.TrimStart('v')

$script:UvExecutable = Resolve-UvExecutable
if ($null -eq $script:UvExecutable) {
    if ($SkipUvInstall) {
        throw 'uv was not found and -SkipUvInstall was supplied.'
    }

    Write-Step 'uv was not found; downloading the official uv installer.'
    $tempInstaller = Join-Path ([IO.Path]::GetTempPath()) 'rookie-agent-uv-install.ps1'
    try {
        Invoke-WebRequest -Uri 'https://astral.sh/uv/install.ps1' -OutFile $tempInstaller
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $tempInstaller
        if ($LASTEXITCODE -ne 0) {
            throw "uv installation failed with exit code $LASTEXITCODE."
        }
    }
    finally {
        if (Test-Path -LiteralPath $tempInstaller) {
            Remove-Item -LiteralPath $tempInstaller -Force
        }
    }

    $script:UvExecutable = Resolve-UvExecutable
    if ($null -eq $script:UvExecutable) {
        throw 'uv installation completed, but uv.exe could not be located. Open a new terminal and run the installer again.'
    }
}

$toolBin = (& $script:UvExecutable tool dir --bin).Trim()
if ($LASTEXITCODE -ne 0) {
    throw 'Could not locate the uv tool executable directory.'
}

$expectedCommandPath = [IO.Path]::GetFullPath((Join-Path $toolBin 'my-agent.exe'))
$resolvedCommands = @(Get-Command my-agent -All -ErrorAction SilentlyContinue)
foreach ($command in $resolvedCommands) {
    if ($command.CommandType -eq 'Application') {
        $resolvedPath = [IO.Path]::GetFullPath($command.Source)
        if ($resolvedPath -ne $expectedCommandPath) {
            throw "A non-uv my-agent command already exists at $resolvedPath. Resolve that PATH conflict before installing Rookie Agent."
        }
    }
}

$rookieState = Get-UvToolState -Name 'rookie-agent'
$legacyState = Get-UvToolState -Name 'my-agent'

if ($null -ne $rookieState -and $rookieState.Editable) {
    throw "Rookie Agent is installed in editable development mode at $($rookieState.EnvironmentPath). The public installer will not overwrite a development environment. Remove it explicitly with 'uv tool uninstall rookie-agent' before using this installer."
}

if ($null -ne $legacyState -and $legacyState.Editable) {
    throw "The legacy my-agent package is installed in editable development mode at $($legacyState.EnvironmentPath). The public installer will not overwrite it. Migrate that development installation manually."
}

if ($null -ne $legacyState) {
    Write-Step "A legacy my-agent $($legacyState.Version) installation was found; preparing a safe migration."

    try {
        Write-Step 'Preflighting the new tagged Rookie Agent package before removing the legacy tool.'
        Install-RookieAgent -Source $source

        Write-Step 'Removing the legacy uv tool after the new package installed successfully.'
        Invoke-Uv tool uninstall my-agent

        Write-Step 'Restoring command entry points from the Rookie Agent package.'
        Install-RookieAgent -Source $source
    }
    catch {
        $migrationError = $_
        $restored = Restore-UvTool -State $legacyState
        if ($restored) {
            throw "Migration failed, and the legacy my-agent installation was restored. Cause: $($migrationError.Exception.Message)"
        }
        throw "Migration failed and automatic legacy restoration was not successful. Cause: $($migrationError.Exception.Message)"
    }
}
else {
    Write-Step "Installing Rookie Agent $Version from $repository."
    try {
        Install-RookieAgent -Source $source
    }
    catch {
        $installError = $_
        if ($null -ne $rookieState) {
            $restored = Restore-UvTool -State $rookieState
            if ($restored) {
                throw "Upgrade failed, and the previous Rookie Agent installation was restored. Cause: $($installError.Exception.Message)"
            }
        }
        throw
    }
}

if ($SkipPathUpdate) {
    Write-Step 'Skipping the persistent PATH update for this isolated installation.'
}
else {
    Write-Step 'Updating the user PATH for uv tools.'
    Invoke-Uv tool update-shell
}

$installedState = Get-UvToolState -Name 'rookie-agent'
if ($null -eq $installedState) {
    throw 'Rookie Agent was not present in the uv tool registry after installation.'
}

if ($installedState.Editable) {
    throw 'The installed Rookie Agent unexpectedly uses editable mode.'
}

if ($installedState.Version -ne $expectedVersion) {
    throw "Expected Rookie Agent $expectedVersion, but uv reports $($installedState.Version)."
}

if (Test-Path -LiteralPath $expectedCommandPath) {
    Write-Step 'Verifying the installed command.'
    & $expectedCommandPath --version
    if ($LASTEXITCODE -ne 0) {
        throw "my-agent --version failed with exit code $LASTEXITCODE."
    }
}
else {
    throw "Rookie Agent installed without creating the expected command at $expectedCommandPath."
}

Write-Host ''
Write-Host 'Installation complete. Open a new PowerShell window, then run: my-agent'
Write-Host 'The first launch will ask for the DeepSeek API key and model inside the TUI.'
