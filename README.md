# Variational Autoencoder in PyTorch

### Usage 

The basic structure of a command looks like:

```
python main.py --mode <mode> --batch_size <num> --lr <learning_rate> 
```

Here `mode` can either be 'train', 'test' or 'inference'. MNIST dataset is automatically downloaded and used. In case a model has been trained before, using `--load` will load this trained model (stored in "TrainedModel/").

