# prostNfoundBK
Reproduction of ProstNFound on BK data

To reproduce Mo's experiments (ProstNFound training on BK data with queen's and UBC) we run: 

Train: 
```
python train_prostnfound_bk.py --exp_dir logs/bk_prostnfound_reprod --warmup_epochs=5 --submitit --time=480 --use_wandb
```

Test: 
```
python main_prostnfound_bk.py test -o test -m logs/bk_prostnfound_reprod/checkpoints/state.pt -c logs/bk_prostnfound_reprod/config.json 
```
