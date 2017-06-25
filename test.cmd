:: test.cmd
:: Author: Michael Friedman
::
:: Runs all tests for the project.

@echo off

:: Validate usage
set usage=test.cmd
if not "%1" == "" (
  echo %usage%
  exit
)

python -m unittest tests
