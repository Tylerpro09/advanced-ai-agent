param(
    [string]$RepoName = "advanced-ai-agent",
    [switch]$Public
)

$ErrorActionPreference = "Stop"
$Visibility = if ($Public) { "public" } else { "private" }

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required."
}
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI (gh) is required: https://cli.github.com/"
}

gh auth status | Out-Null
if ($LASTEXITCODE -ne 0) { throw "GitHub CLI is not authenticated. Run: gh auth login" }

if (-not (Test-Path .git)) {
    git init -b main
    if ($LASTEXITCODE -ne 0) { throw "git init failed." }
}

$Candidates = git ls-files --cached --others --exclude-standard
$SecretPattern = '(^|[\\/])(\.env$|.*\.pem$|.*\.key$|id_rsa$|id_ed25519$)'
if ($Candidates | Where-Object { $_ -match $SecretPattern }) {
    throw "Refusing to publish: a potential secret file is not ignored."
}

git add .
if ($LASTEXITCODE -ne 0) { throw "git add failed." }

git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    git commit -m "Release Advanced AI Agent v2.5"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed. Configure git user.name/user.email if needed." }
}

$remote = git remote get-url origin 2>$null
if ($LASTEXITCODE -eq 0 -and $remote) {
    git push -u origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw "git push failed." }
} else {
    gh repo create $RepoName "--$Visibility" --source=. --remote=origin --push
    if ($LASTEXITCODE -ne 0) { throw "gh repo create failed." }
}

Write-Host "Published: $RepoName ($Visibility)"
