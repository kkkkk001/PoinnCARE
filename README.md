# PoinnCARE

This is the code repository for the ICLR 2026 paper ["PoinnCARE: Hyperbolic Multi-Modal Learning for Enzyme Classification"](https://openreview.net/forum?id=dGxAYNK6JU). 


Experiments in this paper are conducted in the environment specified in `requirements.txt`.
Replicate the experiments using the following instructions:
```
conda create --name poinncare -c conda-forge -c pytorch -c nvidia --file requirements.txt

conda activate poinncare
```

Download the dataset from [Zenodo](https://zenodo.org/records/18813316?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjYzNTg1OWQyLTY2MTEtNGIyMy05ZTI2LTI1NzA1NmY0NzE4YSIsImRhdGEiOnt9LCJyYW5kb20iOiJlNTU2MWU1Y2UxM2IyNjJiNGIxMmU1NDNjYzAxZWQ1YSJ9.m8mJjfhwVC6CgJTJ4LxNZI46woAWq5AgVwLLD98xkiYxqvme2iu4_5yscB-0CKm3fzlBXv4jMUtRVykAwrI3Vw) place it in the `data/` directory.

Then run the following command to predict the EC number:

```
source set_env.sh

python train_align.py --preprocess-adj2 PPR --ppr-alpha2 0.8 --diff-layers2 2 --loss3_weight 0.001 --output1_weight 0.8 --output2_weight 0.2 --save-dir logs/30/ 
```


This project is licensed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) License.
