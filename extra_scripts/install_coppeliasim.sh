#!/bin/bash
# Install CoppeliaSim 4.1.0 for Ubuntu 20.04
# Refer to PyRep README for other versions
export COPPELIASIM_ROOT=${HOME}/.local/bin/CoppeliaSim
wget https://www.coppeliarobotics.com/files/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
## Add environment variables into bashrc (or zshrc)
echo "export COPPELIASIM_ROOT=$COPPELIASIM_ROOT
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT" >> ~/.bashrc
source ~/.bashrc
