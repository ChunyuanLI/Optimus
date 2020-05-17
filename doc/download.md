

## Pre-trained Models
We provide a series of pre-trained *Optimus* models of for different purpose, due to a trade-off between reconstruction capacity and prior regularization.

```bash
wget https://textae.blob.core.windows.net/optimus/$MODEL_DIR/$MODEL_NAME.zip
unzip $MODEL_NAME.zip -d $MODEL_NAME
```
`MODEL_DIR` and `MODEL_NAME` could be different values. We currently release the following models;

This model shows good latent space manipulation performance on SNLI dataset. 
```bash

wget https://textae.blob.core.windows.net/optimus/output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted/checkpoint-full-31250.zip

mkdir output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted
mv checkpoint-full-31250.zip output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted
cd output/LM/Snli/768/philly_vae_snli_b1.0_d5_r00.5_ra0.25_length_weighted

unzip checkpoint-full-31250.zip -d checkpoint-full-31250
```

