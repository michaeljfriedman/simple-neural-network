:: configure.cmd
:: Author: Michael Friedman
::
:: Script to automate installation of dependencies and other configuration of
:: your environment for this project.

@echo off

:: Check for help command
set usage0=usage:
set usage1=configure.cmd
set usage2=configure.cmd --help

set help_message0=This script automates installation of dependencies and other configuration
set help_message1=of your environment for the project.
set help_message2=Arguments
set help_message3=--help          display this message

if "%1" == "--help" (
  echo %usage0%
  echo %usage1%
  echo %usage2%
  echo.
  echo %help_message0%
  echo %help_message1%
  echo.
  echo %help_message2%
  echo %help_message3%
  exit /b
)

:: Validate usage
if not "%1" == "" (
  echo %usage0%
  echo %usage1%
  echo %usage2%
  exit /b
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
