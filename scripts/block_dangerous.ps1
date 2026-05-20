# Blocks accidental commits of large/secret/raw-data files.
$dangerous = @(
    "data/raw/train_FD001.txt",
    "data/raw/test_FD001.txt",
    "data/raw/RUL_FD001.txt",
    "data/raw/*.zip",
    "models/*.pt",
    "models/*.pkl",
    "models/*.json",
    ".env",
    "mlruns",
    "CLAUDE.md",       # private — rule C30
    "MANUAL_TASKS.md", # private — rule C30
    "learning_log.md"  # private — rule C30
)
$staged = git diff --cached --name-only
foreach ($f in $staged) {
    foreach ($pat in $dangerous) {
        if ($f -like $pat) {
            Write-Host "Refusing to commit dangerous path: $f"
            exit 1
        }
    }
}
exit 0
