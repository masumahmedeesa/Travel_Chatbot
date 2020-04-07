# Use following commands into Terminal or Command Prompt to create conda Environment
1. conda create -n newChatbot python=3.5.4 anaconda (with latest version of anaconda)
2. source activate newChatbot [If windows, then activate newChatbot]
3. pip install tensorflow==1.0.0

# To know the installed tensorflow version
pip list | grep tensorflow

# Dataset
1. Cornell movie dialogue corpus

# TO_REMOVE_ENV
conda remove --name newChatbot --all

# INFOS
conda info --envs
