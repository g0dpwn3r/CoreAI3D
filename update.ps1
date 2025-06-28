$rootPath = Get-Location

Get-ChildItem -Path $rootPath -Recurse -Filter "CMakeLists.txt" | ForEach-Object {
    $file = $_.FullName
    Write-Host "Processing: $file"

    $content = Get-Content $file

    $newContent = $content | ForEach-Object {
        if ($_ -match '^\s*cmake_minimum_required\s*\(VERSION\s+[\d.]+\s*\)') {
            'cmake_minimum_required(VERSION 3.20)'
        }
        elseif ($_ -match '^\s*cmake_policy\s*\(VERSION\s+[\d.]+\s*\)') {
            # Skip cmake_policy lines (do not output them)
        }
        else {
            $_
        }
    }

    # Save the modified file
    Set-Content -Path $file -Value $newContent -Encoding UTF8
}
