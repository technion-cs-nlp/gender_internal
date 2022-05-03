# Bias in Bios - training and testing

This folder contains the following code:

1. Train and test RoBERTa and DeBERTa based models on the classification task, with and without fine-tuning.

2. Extract representations of biographies from the last layer of language model.


## Checkpoints

Can be found [here](https://technionmail-my.sharepoint.com/:f:/g/personal/orgad_hadas_campus_technion_ac_il/EpyJD7gHh0tIlVCarqXRdQEBevARdBfa9j7Vo5hgZN_srw?e=qv7Gko).

## Train a model

Use the scripts:

* ``BiasInBios_linear_train.py`` to train a model without fine-tuning. Use ``BiasInBios_extract_vectors.py`` first to peform vectorization to the dataset.
* ``BiasInBios_finetuning_train.py`` with finetuning. Use ``BiasInBios_extract_tokens.py`` first to perform the tokenization to the dataset.
* ``BiasInBios_replace_top_layer.py`` to re-train a model's linear layer while keeping the embedding layer frozen. 
Use ``BiasInBios_extract_vectors.py`` first to extrat the representations from the trained model.

## Test a model

Use the scripts:

* ``BiasInBios_linear_test.py`` to train a model without fine-tuning.
* ``BiasInBios_finetuning_test.py`` with finetuning.
* ``BiasInBios_replace_top_layer_test.py`` to re-train a model's linear layer while keeping the embedding layer frozen.

## Extract vector representations from a model

Use ``BiasInBios_extract_vectors.py``, to extract the vectors for the MDL probing experiments.