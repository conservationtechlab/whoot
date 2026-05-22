Tools to assess BirdNET performance on Buowset

Segments must be a minimum of 3s in order to be assessed by Birdnet, buowsetv1.0
needs padding but buowsetv1.1 are 3s without artificial padding. And then run 
BirdNET analyze over the entire dataset with the desired confidence
thresholding and with burrowing owl as the only class in the species list.

To run Birdnet over your dataset, follow the instructions in this repo:
https://birdnet-team.github.io/BirdNET-Analyzer/usage/cli.html#birdnet-analyzer-analyze

We created our own class list with only our species of interest and ran
birdnet_analyzer.analyze over the entire dataset, beginning with default
confidence and sensitivity values. You can adjust these values and rerun 
to obtain a comparison of performance across different confidence thresholds
and sensitivity. Birdnet will give you a text file result for each audio file
in your dataset, we had these text files saved to the same directory as the
audio.

Running aggregate_birdnet_buowset.py with the path to the BirdNET results and
a .pkl file to send the result to will create a dataframe with the name of the
wav file and a 0 for no buow and a 1 for yes buow detected by BirdNET. 

Then running buowset_assess_birdnet.py with the aforementioned .pkl, the
metadata file for buowset, and some optional paramters, you can compare the
performance of BirdNET against the ground truth labels of buowset. By adding
the optional arguments, you go from comparing BirdNET as a burrowing owl/
no burrowing owl detector to assessing the BirdNET performance on a class by
class basis. If you select to assess for the 'Coocoo' class for example, 
it will aggregate all coocoo instances based on ground truth, and obtain
an equal amount of randomly selected no_buow samples, and generate a confusion
matrix comparing if BirdNET marked the instances of that class as burrowing owl. 

Because BirdNET is a binary classifier for burrowing owl in this data, a class
by class comparison only tells us if it disproportionately misses certain calls
more than others when looking for burrowing owls in general, ie it gives us a peak
into the likely call distribution of their training data.
