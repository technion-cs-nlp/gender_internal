# Coreference resolution - training and testing

This folder contains code that is based on the repo from the work by [Xu and Choi (2020)](https://github.com/emorynlp/coref-hoi).

## Data

To train the model, you need to get Ontonotes 5.0 CONLL files, and then use the script ``setup_data.sh`` to compile these
files to the format the script works with. Same goes for Winobias CONLL files for the evaluation part. See more info in the ``data``
folder of this repo.

## Environment installation

The environment training and (part of) testing the model is different from the rest of the repo, because we were based on [Xu and Choi (2020)](https://github.com/emorynlp/coref-hoi). The environment installation is given by the requirements.txt file.

Use:

        pip install requirements.txt

Or, if you're using conda to manage your environments, create a new environment and install the packages one-by-one.

Some of the other test scripts, written by us, requires the original environment. We describe which environment is needed for each one.

## Checkpoints

Checkpoints can be found [here](https://technionmail-my.sharepoint.com/:f:/g/personal/orgad_hadas_campus_technion_ac_il/EoXz9nITf8RLjHupqrHyP_YB-9ew-pPOBUVS1mXIrfP4oQ?e=3kVXkT). Due to the size of the files, we split it to 5 files: all the models in the original training,
and the re-trained models per each training method. For example, ```coref_checkpoints_retrain_anon.zip``` contains all of the
re-trained models, where the embedding was trained before on anonymized data.

## Training a coreference resolution model

###TL;DR:

to train a model, use the following commands:

        python run.py experiment_name gpu_id seed

For the retraining experiment, run:

        python run.py experiment_name gpu_id seed embedding_name
        
examples:

```python run.py roberta_base 0 0``` - to train a model without debiasing, using gpu 0 and seed 0.

```python run.py roberta_base_balanced 0 0``` - to train a model anonymization and CA.

```python run.py roberta_base_anonymized 0 0``` - to train a model with anonymization only.

```python run.py roberta_base_CA 0 0``` - to train a model with CA only.

```python run.py roberta_base_CA 0 0 CA``` - to re-train a model with embeddings that were trained on CA data. Note that
for retraining you first need to extract the embedding model using the script ```extract_embedding_model.py```.

### Explanation:

To train a model, use ```experiments.conf``` file. The following experiments are used for training:

```roberta_base```: used for training a model with roberta_base (from huggingface) as a feature extractor, without any debiasing strategy.

```roberta_base_balanced```: used for training the CA+Anon model on the data (counterfactual augmentation and anonymization debiasing strategies).

```roberta_base_anonymized```: only anonymization.

```roberta_base_CA```: Only CA (counterfactual augmentation).

The experiments are ready to run out-of-the-box, and will create folders with the training logs and models' checkpoints.
The path for the models' checkpoints is in the field ```model_path```.

If you are interested in exploring other models besides roberta_base, you can change the configuration accordinly, but note that it has
to be a model the can be loadede using Huggingface's function ```AutoModel.from_pretrained```, or you'll need to change 
the code accordingly (in ```model.py```).

## Testing the model

### Ontonotes

Ontonotes is used to test that the model actually learned the task, but not for measuring gender bias.
To run the tests:

        python run.py experiment_name model_id gpu_id

Where ```experiment name``` is one of the test experiments, such as test_roberta_base. The experiment name is used to specify
the test file. ```model_id``` refers to a file in the form of ```model_{model_id}.bin```, in the folder specified byh 
```model_path``` in the experiment.

### Winobias

The following functions require the environment of the original project, not the coreference project:
* ``winobias_other_metrics.py``
* ``winobias_F1.py``

There are several scripts to test data on [Winobias](https://arxiv.org/abs/1804.06876).

Use ``winobias_F1.py`` to compute the F1 diff computed in the original Winobias paper.
Use ``get_predictions.py`` to generate a predictions file to then be fed to ``winobias_other_metrics.py``, 
which requires the environment of the original project, not the coreference environment.

## Computing compression (MDL probing)

First extract vector representations for Winobias using ``Winobias_extract_vectors.py``.

Then, use the script ```Winobias_MDL_probing.py```, with the original project's environment.