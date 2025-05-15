#!/bin/bash

# Set a simplified bash prompt
echo "export PS1='\u:\w\$(__git_ps1 \" (%s)\")\$ '" >> /root/.bashrc

# Source the updated .bashrc to apply changes
source /root/.bashrc
