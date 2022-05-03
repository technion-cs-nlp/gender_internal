# How Gender Debiasing Affects Internal Model Representations, and Why It Matters 

This repo contains the code for reproducing the results in the paper.

The repo is organized as follows:

1. **bio** - contains the code to train and test the models on Bias in Bios task.

2. **coref** - contains the code to train and test the models on coreference resolution task (Ontonotes and Winobias).

3. **compression** - contains the code to measure the compression rate of gender information in a model, using MDL probes.

4. **CEAT** - contains the code to measure the metric CEAT in models.

Each folder contains its own documentation, and we provide links to the data (or ways to reproduce it) and links to model checkpoints.
The code of the experiments contains code to automatically log the results to [Wandb](https://wandb.ai/). All of the scripts
used in this repo are each well documented (use ``name_of_script.py`` -h).

We use ``Wandb`` to log the experiment's results. If you're not interested, comment out the relevant lines of code (search for "wandb"
in the files to find the lines to comment out).

## Environment installation

The requirements.txt file is for creating a conda environment.
This environment will be used for most scripts, but some of the scripts for coreference resolution require a different
environment, as we based the code on another work.

        conda create --name genderbias --file requirements.txt
        conda activate genderbias
        
## Contact us
For questions, please reach out to the authors of the paper.