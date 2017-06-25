:: configure.cmd
:: Author: Michael Friedman
::
:: Script to automate installation of dependencies and other configuration of
:: your environment for this project.

:: Some preliminary configuration
@echo off
setlocal EnableDelayedExpansion
set NL=^


:: Note the two blank lines above are required to set the NL variable

:: Check for help command
set usage=usage:!NL!configure.cmd!NL!configure.cmd --help

set help_message0=This script automates installation of dependencies and other configuration
set help_message1=of your environment for the project!NL!
set help_message2=Arguments
set help_message3=--help          display this message

if "%1" == "--help" (
  echo %usage%
  echo
  echo %help_message0%
  echo %help_message1%
  echo %help_message2%
  echo %help_message3%
  exit
)

:: Validate usage
if not "%1" == "" (
  echo %usage%
  exit
)

::------------------------------------------------------------------------------

:: Configure environment

:: Install Python dependencies to virtual envionment
if not exist venv (
  echo ----- Creating Python virtual environment -----
  virtualenv venv
  echo ----- Done creating Python virtual environment -----
)

venv\Scripts\activate

echo ----- Installing Python packages -----
pip install numpy
echo ----- Done installing Python packages -----

deactivate

echo Remember to activate the virtual environment with 'venv\Scripts\activate' to use the Python packages.
