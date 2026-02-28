# PoinnCARE

This is the code repository for the ICLR 2026 paper ["PoinnCARE: Hyperbolic Multi-Modal Learning for Enzyme Classification"](https://openreview.net/forum?id=dGxAYNK6JU). 


Experiments in this paper are conducted in the environment specified in `requirements.txt`.
Replicate the experiments using the following instructions:
```
conda create --name poinncare -c conda-forge -c pytorch -c nvidia --file requirements.txt

conda activate poinncare
```

Then run the following command to predict the EC number:

```
source set_env.sh

python train_align.py --preprocess-adj2 PPR --ppr-alpha2 0.8 --diff-layers2 2 --loss3_weight 0.001 --output1_weight 0.8 --output2_weight 0.2 --save-dir logs/30/ 
```


This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.
