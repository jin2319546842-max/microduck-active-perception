$demoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $demoRoot
& "$demoRoot\.venv\Scripts\python.exe" "$demoRoot\demo.py"
