# Fine-tuning Optimus on a VAE language modeling task

_Note: The latent vector size has a great impact on the model performance: small latent size provides a tight information bottleneck, often yielding low reconstruction quality; In contrast, large latent size shows good reconstruction quality, but would be hard to control the latent manipulation with vector operators if the latent size is too large. To have a fair comparison with existing works, we use a 32-dimensional latent vector for the experiments on language modeling._ 

##### Download a pre-trained model (pre-trained from Wikipedia). 
