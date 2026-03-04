In order to easily upload data to edge-impulse for model training, 
we will physically organize the folder structure of our data into

```
.
├── train
│   ├── bird_of_interest
│   ├── noise
├── test
    ├── bird_of_interest
    └── noise
```

Where train is 80% (4 folds) of the data, split into your two classes
in a subfolder. And test is 20% (1 folds worth) split in the same way.

This way, you can just upload each folder and specifiy test/train and
what class all the samples belong to without needing a metadata file. 
