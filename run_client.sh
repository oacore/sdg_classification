#!/bin/bash

# Get the directory path of the current script
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


# Set PYTHONPATH environment variable
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"


python3 "$PROJECT_DIR/src/client.py"