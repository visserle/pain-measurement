
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
Write-Host "Starting..."

# Execute Python scripts
python -m src.models.main --features eda_raw --models MLP LightTS TimesNet PatchTST iTransformer
python -m src.models.main --features heart_rate --models MLP LightTS TimesNet PatchTST iTransformer
python -m src.models.main --features pupil --models MLP LightTS TimesNet PatchTST iTransformer
python -m src.models.main --features eda_raw heart_rate --models MLP LightTS TimesNet PatchTST iTransformer
python -m src.models.main --features eda_raw pupil --models MLP LightTS TimesNet PatchTST iTransformer
python -m src.models.main --features eda_raw heart_rate pupil --models MLP LightTS TimesNet PatchTST iTransformer
python -m src.models.main --features face --models MLP LightTS TimesNet PatchTST iTransformer
python -m src.models.main --features eeg --models MLP LightTS iTransformer EEGNet
python -m src.models.main --features eeg eda_raw --models MLP LightTS iTransformer EEGPhysioEnsemble
python -m src.models.main --features eeg face eda_raw heart_rate pupil --models MLP LightTS iTransformer EEGFacePhysioEnsemble

Write-Host "Model comparison completed."
Write-Host ""
Write-Host "Press [Enter] to exit..."
Read-Host
exit