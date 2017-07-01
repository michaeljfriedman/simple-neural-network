# test
# Author: Michael Friedman
#
# Runs all tests for the project.

# Validate usage
$usage="Usage: .\test"
if ($args.Length -ne 0) {
  Write-Host "$usage"
  exit
}

# Run tests
python -m unittest tests
