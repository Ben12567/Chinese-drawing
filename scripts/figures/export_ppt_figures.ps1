# Render outputs/revision/figures_ppt/FigN.pptx to FigN.png (2250 px wide = 7.5 in at 300 dpi) with PowerPoint.
param([string]$Dir = (Join-Path $PSScriptRoot "..\..\outputs\revision\figures_ppt"))
$Dir = (Resolve-Path $Dir).Path
$pp = New-Object -ComObject PowerPoint.Application
try {
    foreach ($f in Get-ChildItem -Path $Dir -Filter "Fig*.pptx") {
        $pres = $pp.Presentations.Open($f.FullName, $true, $false, $false)
        $w = 2250
        $h = [int][math]::Round($w * $pres.PageSetup.SlideHeight / $pres.PageSetup.SlideWidth)
        $out = Join-Path $Dir ($f.BaseName + ".png")
        $pres.Slides.Item(1).Export($out, "PNG", $w, $h)
        $pres.Close()
        Write-Output "$($f.Name) -> $out ($w x $h)"
    }
} finally {
    $pp.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp) | Out-Null
}
