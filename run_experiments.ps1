# Check for PowerShell version 6.0 or higher
if ($PSVersionTable.PSVersion.Major -lt 6) {
    Write-Host "Error: This script requires PowerShell version 6.0 or higher."
    Read-Host
    exit
}

# Initialize Conda environment
if ($IsWindows) {
    # Windows specific Conda initialization
    $condaPath = "C:\mamba"
    & "$condaPath\shell\condabin\conda-hook.ps1"
    # Depending on the Conda setup you might need this instead to initialize Conda
    # & "$condaPath\Scripts\activate.ps1"
    conda activate pain
} elseif ($IsMacOS) {
    # macOS specific Conda initialization (only used for development without iMotions and the respective dummy_imotions flag)
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
Write-Host "This is the behavioral pain measurement experiment."
Write-Host ""
Write-Host ""
Write-Host "The experiment consists of four parts:"
Write-Host "1. Pre-Experiment Questionnaires"
Write-Host "2. Pain Calibration"
Write-Host "3. Pain Measurement"
Write-Host "4. Post-Experiment Questionnaires"
Write-Host ""
Write-Host "Instructions:"
Write-Host "- Please make sure that the participant is ready and seated comfortably."
Write-Host "- After the completion of each part, the program will pause and wait for you to press [Enter] to continue."
Write-Host "- Note: If you need to abort the experiment, press [Ctrl+C]."
Write-Host ""
Write-Host ""
Write-Host "Press [Enter] to start..."
Read-Host

# Execute Python scripts
Write-Host "1. Pre-Experiment Questionnaires"
python -m src.experiments.participant_data  # add a new participant
python -m src.experiments.questionnaires.app general panas
Write-Host "Press [Enter] to continue with the calibration..."
Read-Host

Write-Host "2. Pain Calibration"
python -m src.experiments.calibration.calibration
Write-Host "Press [Enter] to continue with the measurement..."
Read-Host

Write-Host "3. Pain Measurement"
python -m src.experiments.measurement.measurement
Write-Host "Press [Enter] to continue with the questionnaires..."
Read-Host

Write-Host "4. Post-Experiment Questionnaires"
python -m src.experiments.questionnaires.app panas maia-2 pcs pvaq lot-r bdi-ii brs stai-t-10 erq maas

# Print the completion message
Write-Host "Experiment completed."
Write-Host ""
Write-Host "╔═════════════════════════════════════════════════════════════════════════════╗"
Write-Host "║                                                                             ║"
Write-Host "║                                                                             ║"
Write-Host "║                                                                             ║"
Write-Host "║    ██████╗  ██████╗  ██████╗ ██████╗          ██╗ ██████╗ ██████╗     ██╗   ║"
Write-Host "║   ██╔════╝ ██╔═══██╗██╔═══██╗██╔══██╗         ██║██╔═══██╗██╔══██╗    ██║   ║"
Write-Host "║   ██║  ███╗██║   ██║██║   ██║██║  ██║         ██║██║   ██║██████╔╝    ██║   ║"
Write-Host "║   ██║   ██║██║   ██║██║   ██║██║  ██║    ██   ██║██║   ██║██╔══██╗    ╚═╝   ║"
Write-Host "║   ╚██████╔╝╚██████╔╝╚██████╔╝██████╔╝    ╚█████╔╝╚██████╔╝██████╔╝    ██╗   ║"
Write-Host "║    ╚═════╝  ╚═════╝  ╚═════╝ ╚═════╝      ╚════╝  ╚═════╝ ╚═════╝     ╚═╝   ║"
Write-Host "║                                                                             ║"
Write-Host "║                                                                             ║"
Write-Host "║                                                                             ║"
Write-Host "╚═════════════════════════════════════════════════════════════════════════════╝"
Read-Host
