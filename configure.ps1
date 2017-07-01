# configure.ps1
# Author: Michael Friedman
#
# Script to automate installation of dependencies and other configuration of
# your environment for this project.

# Check for help command
$usage = @"
Usage: .\configure
       .\configure --help
"@

$help_message = @"
This script automates installation of dependencies and other configuration
of your environment for the project.

Arguments
--help          display this message
"@

if (($args.Length -eq 1) -and ($args[0] -eq "--help")) {
  Write-Host $usage
  Write-Host
  Write-Host $help_message
  exit
}

# Validate usage
if ($args.Length -ne 0) {
  Write-Host $usage
  exit
}

#------------------------------------------------------------------------------

# Configure environment

# Install Python dependencies to virtual environment
if (!(Test-Path venv)) {
  Write-Host "----- Creating Python virtual environment -----"
  virtualenv venv
  Write-Host "---- Done creating Python virtual environment -----"
}

.\venv\Scripts\activate.ps1

Write-Host "----- Installing Python packages -----"
pip install numpy
Write-Host "----- Done installing Python packages -----"

deactivate

Write-Host "Remember to activate the virtual environment with '.\venv\Scripts\activate' to use the Python packages."

