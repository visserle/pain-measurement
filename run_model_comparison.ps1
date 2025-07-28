
# Check for PowerShell version 6.0 or higher
if ($PSVersionTable.PSVersion.Major -lt 6) {
    Write-Host "Please install PowerShell 6.0 or higher to run this script."
    Read-Host
    exit
}

# Initialize Conda environment
if ($IsWindows) {
    # Windows specific Conda initialization
    $condaPath = Join-Path $env:USERPROFILE "miniconda3"  # Dynamically find miniconda3 in user's home directory
    & "$condaPath\shell\condabin\conda-hook.ps1"
    # Depending on the Conda setup you might need this instead to initialize Conda
    # & "$condaPath\Scripts\activate.ps1"
    conda activate pain
}
elseif ($IsMacOS) {
    # macOS specific Conda initialization
    $condaPath = "$HOME/miniforge3"
    # Use bash to activate the environment and get the Python path
    $pythonPath = /bin/bash -c "source '$condaPath/bin/activate' pain; which python"
    
    # Extract the directory path of the Python executable
    $pythonDir = Split-Path -Parent $pythonPath
    
    # Prepend the Python directory to the PATH environment variable
    $env:PATH = "$pythonDir" + ":" + $env:PATH
}

# Print the welcome message
Write-Host ""
Write-Host "This is the model comparison script for different feature combinations."
Write-Host ""
Write-Host ""
Write-Host "Press [Enter] to start..."
Read-Host

# Execute Python scripts
python -m src.models.main --features eda_tonic --models MLP TimesNet Crossformer PatchTST 
python -m src.models.main --features eda_phasic --models MLP TimesNet Crossformer PatchTST 
python -m src.models.main --features eda_tonic eda_phasic --models MLP TimesNet Crossformer PatchTST 
python -m src.models.main --features pupil_mean --models MLP TimesNet Crossformer PatchTST 
python -m src.models.main --features heartrate --models MLP TimesNet Crossformer PatchTST 
python -m src.models.main --features eda_tonic eda_phasic heartrate --models MLP TimesNet Crossformer PatchTST 
python -m src.models.main --features eda_tonic eda_phasic heartrate pupil_mean --models MLP TimesNet Crossformer PatchTST 
python -m src.models.main --features brow_furrow cheek_raise mouth_open upper_lip_raise nose_wrinkle --models MLP TimesNet Crossformer PatchTST 
python -m src.models.main --features eeg --models MLP TimesNet Crossformer PatchTST 

Write-Host "Model comparison completed."
Write-Host ""
Write-Host "Press [Enter] to exit..."
Read-Host
exit