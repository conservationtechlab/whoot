# Software Evaluations for Annotating Audio Data
## Software options that create .csv files with label, start_time, end_time, bounding boxes, and/or whether they can add multiple labels to a single audio file:

- [Label Studio](https://labelstud.io) (Prefered Option, #1)
- [RavenPro](https://www.ravensoundsoftware.com/software/raven-pro/) (#2)
- [Kaleidoscope](https://www.wildlifeacoustics.com/products/kaleidoscope-pro) (Free or Pro?) (#3)
- [PAMGuard](https://www.pamguard.org/) (#4)
- [ARBIMON](http://arbimon.org/)
- [Pyrenote-desk](https://github.com/UCSD-E4E/pyrenote-desk)
  - [Wavesurfer.js](https://github.com/katspaugh/wavesurfer.js)
- [Audacity](https://www.audacityteam.org/)
- [Avisoft-SASLab](https://avisoft.com/sound-analysis/)

* The original search was for annotating bat sound data, so we have included information specifically for bats. Please disregard unless your animal call is greater than 20kHz.

## Notes on each software: 
### Label Studio (Free version): 
__(No longer considering for bats; Decided at Whoot Meeting on 1-13-2026)__
*Based on the current formating of labelstudio, a team would need to develope and code a labelstudio template or bat version before we can proceed.
  - __Pros:__
    - Web-based software
    - User-friendly GUI
    - Can listen to audio file and see oscillogram and spectrogram
    - Unlimited number of labels per project
    - Can add multiple labels to one audio recording
    - Labeling is simple and straightforward; specify start and end time of audio section. Don't see a bounding box style annotation capabilities
    - Can adjust spectrogram parameters within the software code (not the GUI)?
    - Open source program
  - __Cons:__
    - Recommended to clip larger wav files to 5-10 minutes (easier for users; can handle larger files, but creates long loading times and many labels to mark)
    - Cannot hear bat sounds (>20kHz), even when playback speed is set to half. (On Kaleidoscope, playback is at 1/16 or 1/8 to hear bat calls)
    - All preprocessing must be performed before uploading data to it (cannot change pitch, perform time expansion, or change audio speed (rate))
    - Will need pre-processing to artificially lower the bat calls, but keep the frequency information on the y-axis of the spectrogram for idenification information.
    - Will need to create a new instance setup for Peru folks (is too difficult to organize and provide access to our local labelstudio remotely)
    - Will need to send code to set up spectrogram for viewing (not a default feature in the software)
    - In Label Studio, vertical (y-axis) zooming for spectrograms is not currently a native feature. Zooming is typically restricted to the horizontal (x-axis) timeline.
    
    *Paid version: $149/month, additional users $99/month (up to 12 users); other capabilities are accessible but need to speak with someone about custom pricing

### RavenPro (RavenLite can perform multiple annotations, but cannot adjust spectrogram parameters)
  - __Pros:__
    - Linux and Windows compatible
    - GUI is a little complex, but not difficult to learn; would need to provide a detailed SOP document
    - Visualizes oscillogram and spectrogram
    - Can create multiple labels (start-end time or bounding boxes) on the spectrogram (saves as a .txt file, one table file per recording!)
    - Can perform spectral and sound parameter measurements
    - Apply filters (high pass, low pass, band pass)
    - Can play auto at a slower rate to "hear" the bats
    - Has been used in published literature for manual annotation of bat data, even from AudioMoth ([Brinkløv et. al, 2022](https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1111/2041-210X.14131))
    - Pro version has BirdNet Neural Network built-in for classification of birds
  - __Cons:__ 
    - No preset labels, must type in label (could introduce labeling errors from misspelled labels)
    - Originally designed for bird calls
    - Proprietary software, so we don't know exactly what's going on "under the hood"
    * Pro version costs $100 per license for non-profits, for more information: [RavenPro Pricing](https://www.ravensoundsoftware.com/raven-pricing/)

### Kaleidoscope (Free version)
  - __Pros:__
    - Linux and Windows compatible
    - Designed for bat analysis workflows; Has bat analysis system (has built-in ID option with Pro)
    - Has time expansion for bat data
    - Visualizes oscillogram and spectrogram
    - Can adjust spectrogram parameters using the GUI 
    - Can zoom-in and out of spectrogram to focus on a particular part of the audio file and can play just that highlighted section.
    - Can label with multiple labels
  - __Cons:__
    - GUI is not intuitive, but options are limited in free version; would need to provide a detailed SOP document
    - Can only have 32 labels available at anytime while using the software
    - Labels the entire audio file and does not give start and stop times (This might be better in the pro version...)
    - Will need pre-processiong into smaller chunks (1-5s)
    - Proprietary software, so we don't know exactly what's going on "under the hood"
    * Pro version is costly $399 for software per user + Subscription + Cloud storage and data upload costs. Subscription allows two activations for a single user for 12 months.
    ** Here is a intro to Kaleidoscope video for bat data (https://www.youtube.com/watch?v=eT9XYPpESPw)

### PAMGuard
  - Originally made for underwater "click" detections of cetaceans. Can also process bat data and has pipelines for bats from AudioMoth recorders specifically
  - __Pros:__
    - Free Software!
    - Linux and Windows compatible? 
    - Open source program
    - Has been used in published literature for manual annotation of bat data, even from AudioMoth ([Brinkløv et. al, 2022](https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1111/2041-210X.14131))
    - Visualizes oscillogram and spectrogram
    - Can label calls manually
    - Has a built-in neural network, ANIMAL-SPOT, for event detection and classification of calls
    * Intro to PAMGuard video with bat data: https://www.youtube.com/watch?v=zqwy1daloMY
  - __Cons:__
    - Not intuitive to initially setup, but once module settings are saved, the program will automated load with those modules during future sessions.
    - GUI is not user-friendly; would need to provide a detailed SOP document 

### Audacity 
__(No longer considering for bats; Decided at Whoot Meeting on 1-7-2026)__
  - __Pros:__
    - Linux and Windows compatible
    - Visualizes oscillogram and spectrogram (just have to copy oscillogram to a new track, mute track, and change view to spectrogram.)
    - Can create multiple labels (.txt file) but only based on oscillogram (provides: start, stop, label_name)
    - GUI is fairly easy to operate
  - __Cons:__ 
    - Cannot adjust spectrogram visualization (except for zoom) or how the spectrogram is calculated
    - Proprietary software, so we don't know exactly what's going on "under the hood"

### ARBIMON
__(No longer considering for bats; Decided at Whoot Meeting on 1-13-2026)__

  - Good introduction [video](https://www.youtube.com/watch?v=YRSRdilIVHM) from Carly Batist from the K. Lisa Yang Center for Conservation Bioacoustics YouTube Channel
  - __Pros:__
    - Web-based and cloud-enabled (easy access)
    - Multi-user capability and can assign users different roles/access to each project (can download from cloud as well)
    - Can play audio recordings; can handle high sampling rate (up to 384kHz) AudioMoth data
    - Visualizes the spectrogram of the data [[timestamp: 12:50](https://youtu.be/YRSRdilIVHM&t=770)]
    - Can annotate spectrograms (creates bounding boxes, which includes start and end times and frequency max and min)
    - Can create a predetermined template of all the labels and select them
    - Options are available to keep data.project private
    - Includes tagging audio file (i.e. issues with the recording, no species of interest detected, etc.)
    - Includes "Species Presence Validation", which is a drop-down yes-no menu indicating where certain species are contained in the entire audio file (separate from the labels); can import a "species list"
    - Apply filters (high pass, low pass, band pass)
    - Can adjust the gain of the audio
    - Has Pattern Matching (Cross-correlation algorithm), Random Forest Models, Audio Event Detection, and Soundscape (Energy Density Spectrogram?)
      - Clustering has a cap on the size of the playlist (n=?) that you can run, but should take less than a day to run
      - Can reach to someone at ARBIMON to increase the clustering cap if needed
    - Also includes 10 Regional Classification CNNs [[Timestamp 28:03](https://youtu.be/YRSRdilIVHM&t=1683)]
      - Panama, Edudor, Puerto Rico, Sumatra, Brazil, ...
      - 670 species
      - Can manually validate the outputs
    - Has a public library of labeled spectrogram data for specific calls for 6000+ species (includes, birds, frogs, and bats)
    - Has an "Insites" tab that provides information about hotspots for species richness, number of species vs. hour of day, detection Frequncy, and naive occupancy
    - *Has option to exlude certain animals' location data (i.e. endangered) from public access 
  - __Cons:__
    - Web-based and cloud-enabled (have to upload data to the cloud)
    - Proprietary software, so we don't know exactly what's going on "under the hood"
    - No Visualization of Oscillogram
    - __Ownership of any data uploaded to ARBIMON is in question and will need to be negotiated with ARBIMON!__
    
    *Email for inquires: contact@rfcx.org

  ### Pyrenote-desk (Is not in a functional state to ship out to labelers)
  - Pyrenote creates moment to moment or strong labels for audio data. Pyrenote and much of this README are based on heavily on Audino as well as Wavesurfer.js.
  - __Pros:__
    - Linux compatible
    - UI is simple and can generally follow the flow 
    - Visualizes both the Oscillogram and Spectrogram (but what type is not indicated; would be nice to include different specrogram options.)
    - Can resize and add multiple labels

  - __Cons:__
  *Using Ubuntu 24
    - Difficult to install; installation instructions are not detailed enough, missing packages to install (node.js, npm, curl).
    - Needs better installation notes that do not require links to other tutorials.
    - Ddding recordings requires lots of metadata to be inputted first, and I __cannot__ upload recordings without deployment information (which needs to be optional).
    - `Import all` fails while `Import Selected` with no filters works as an import all function.
    - Oscillorgram and spectrogram doesn't show y-axis information which can be critical to identifying certain species (like bats!). Also there is no option to change the spectrogram color scale. Currently using suboptimal colorscale.
    - Zoom bar does not work.
    - Speed playback bar causes an error that generates multiple new oscillogram and spectrogram tracks
    - Playback at anything other than 1.0 does not work.
    - GUI lags a ton when adding new labels by selecting the labels button in the left-hand menu. Also if too many labels rows/spectrograms are added, the play bar and buttons to remove or undo disappear.
    - Additionally, delete, clear, and undo buttons do not work.
    - When I exit out of the application via the X button in the top right corner, the terminal continues to run unless I hit ctrl+C.

  *This uses websurfter.js, so I have reviewed it as well.
  ### wavesurfer.js
  - Wavesurfer.js is an interactive waveform rendering and audio playback library, perfect for web applications. It leverages modern web technologies to provide a robust and visually engaging audio experience.
  - __Pros:__
    - Customizable: Can create our own labeling code and pipeline
    - Mostly for web-based applications
    - Can visualize oscillogram and spectrograms
    - Can annotate via UI, but would have to code resizing, dragging, and save out features
  
  - __Cons:__
    - Nothing is pre-built, except for functions
    - Can make a web-based or desktop application, but would have to be a finished product before shipping to labelers.

  *I have uploaded errors and my functionality requests for pyrenote-desk to Issues on their github.

## Avisoft-SASLab (Lite/Pro)
  - Originally made for underwater "click" detections of cetaceans. Can also process bat data and has pipelines for bats from AudioMoth recorders specifically
  - __Pros:__
    - Has a Lite free software version 
    - Proprietary software, so we don't know exactly what's going on "under the hood"
    - Windows compatible
    - Visualizes oscillogram and spectrogram (Lite only lets you use n_fft=256 and cannot change this setting)
    - Can adjust playback settings to slow and adjust the pitch to hear the bats.
    - Digital filtering available (High, Low, Bandpass, Bandstop, User-defined, band rejection, and reduction filters)
    - Can perform sound parameter measurements
    - Can perform even detection
    - Can label calls manually but only with Pro
    - Has nice documentation
  - __Cons:__
    - Not Linux compatible
    - GUI is not intuitive (especially in Lite version)
    - Cannot adjust constrast of spectrogram, so it is difficult to see bat calls, unless the call is VERY loud
    - Only labeling option is for single point with time duration.
    - Does not have magma or plasma color scheme for viewing spectrogram
    - Annotations are not available in lite version
    - Pricing for the Pro version is €2900, which is ~$3,400 (on 1/22/26)
