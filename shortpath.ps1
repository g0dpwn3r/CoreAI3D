function Get-ShortPath([string]$path) {
    $obj = New-Object -ComObject Scripting.FileSystemObject
    return $obj.GetFile($path).ShortPath
}

Get-ShortPath "D:\code\CoreAI3D\out\build\windows-x64-debug\CoreAI3D\CoreAI3D.exe"
