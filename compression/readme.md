# MDL Probing

The scripts are well documented, but be aware that you first need to extract internal representations using the scripts:

* ``BiasInBios_extract_vectors.py`` from ``bio`` folder.
* ``Winobias_extract_vectors.py`` from ``coref`` folder.

Then, use either ``BiasInBios_MDL_probing.py`` or ``Winobias_MDL_probing.py``.

Our implementation is based on the implementation from [Debiasing Methods in Natural Language Understanding Make Bias More Accessible
](https://arxiv.org/abs/2109.04095).