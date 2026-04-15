## Label Studio Labeling Interface Code

Label Studio allows users to define their labeling interface
preferences and options using XML. We have prepared some
premade templates to use when labeling various data.

### buow_acoustic_validation.xml

This template assumes start/stop time predictions within (longer segments) 
are uploaded to labelstudio with the intention to validate these
predictions. The labeler will mark whether a prediction is
correct or incorrect, and if incorrect, will be shown options to select
if it is a different vocalization type, something else, or they're not sure.
