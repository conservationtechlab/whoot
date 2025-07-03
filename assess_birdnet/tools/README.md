Older tools to assess birdnet against the human labeled burrowing owl data from 2017-2018.

These tools assess birdnet by splitting all audio into 3s chunks, irregardless of where
a labeled detection occurred, and adds the human labels onto the 3s chunks after this
chunking occurs, if the detection window has ANY overlap with a 3s segment. It then
compares this on an individual wav file basis to the birdnet results for the same data.

We have since moved onto assessing BirdNET on Buowset, the dataset created out of our
human labeled burrowing owl data. 
