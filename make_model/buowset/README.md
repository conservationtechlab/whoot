These scripts are intended to load and format and parse
the buowset dataset to easily train models from differing
embeddings sources. 

Using Birdnet Embeddings:
Birdnet needs each sound clip to be a minimum of 3s long,
so many of the segments need to be padded in order to reach
this minimum. 

1) Running zero_pad_detections.py on the entire dataset will
copy it to a new folder, containing all the same segments in
buowset but padded with silence either added to the end of the 
sample, or randomly dispersed silence at the beginning and end.
You can decide how long the minimum length of each segment should
be, so working with a different embedding source that requires
a different minimum length than Birdnet is still possible, simply
change the length parameter.

2) With your newly padded dataset, run birdnet.embeddings on the
entire folder.

3) Then pass along the directory containing your birdnet embeddings,
your metadata.csv for buowset, and a path for your merged data as
a .pkl and run embed_to_df_birdnet.py. 

4) Pass this .pkl result to make_svm.py, along with an optional path
to a model.pkl file if you'd like to save the model that is
produced. 

You're done!

Optionally, combine steps 3 and 4 into one step by running
make_birdnet_svm.py and passing the birdnet embeddings, metadata,
and optional model save file.
If you are using the same result from step 3 repeatedly and merely
changing model parameters, properly go through steps 3 and 4 so you
just need to repeat step 4.

Defaults: make_svm.py and make_birdnet_svm.py rely on global parameters
defined in make_svm.py. 
TRAINING_FOLDS: Change these numbers to be the folds you wish to go
in your training set. Default is 0-3.
TESTING_FOLDS: Change this number(s) to be the fold(s) you wish to go
in your testing set. Default is 4.
CLASS_0: This SVM pipeline creates binary classifiers. Change the numbers
of this variable to reflect which of the 6 classes in buowset you'd like to
be 0, all unlisted classes will be 1. Default here is that the 'no_buow' class
is the 0, or int 5. This default is also the only balanced distribution for
making a binary SVM with buowset as is. 
