# Software Evaluations for Annotating Audio Data
## Software options that create .txt files with label, start_time, end_time, bounding boxes (includes frequency range), and/or whether they can add multiple labels to a single audio file:

- [ARBIMON](http://arbimon.org/)
- [Audacity](https://www.audacityteam.org/)
- [Avisoft-SASLab](https://avisoft.com/sound-analysis/)
- [Kaleidoscope](https://www.wildlifeacoustics.com/products/kaleidoscope-pro)
- [Label Studio](https://labelstud.io)
- [PAMGuard](https://www.pamguard.org/)
- [Pyrenote-desk](https://github.com/UCSD-E4E/pyrenote-desk)
  - [Wavesurfer.js](https://github.com/katspaugh/wavesurfer.js)
- [RavenPro](https://www.ravensoundsoftware.com/software/raven-pro/)

* The original search was for annotating bat sound data, so we have included information specifically for bats. Please disregard unless your animal call is greater than 20kHz.

## EDITING DOCUMENT INFO
__If you are going to edit this document, always double check that both the table and the written notes have been updated! Thank you!__ 

## Table: Overview of Capabilities for Each Reviewed Software
| Software Name | Cost | Local or Online | OS Platforms | Visualization Capabilities | Spectrogram Adjustable Settings? | Annotation Type | Playback Options? | Filtering Options? | Sound Measurements? | Built-in Algorithms/AI Options? | Multi-User? | Batching Options? | GUI Interpretability Difficulty |
| :---  | :---  | :---  | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [ARBIMON](http://arbimon.org/) | Free, No License | Online, Web/Cloud-based | Windows and Linux |Spectrogram, Energy Density Spectrogram, and more | Yes | Multi, Bounding Boxes | Yes | Yes | Yes | Yes | Yes | Yes | Medium Difficulty, lots of options in different tabs | 
| [Audacity](https://www.audacityteam.org/) | Free, No License | Local | Windows and Linux | Oscillogram, Spectrogram, Power Spectrum | Yes | Multi, Start-stop| Yes | Yes | No | No | No | No | Easy Difficulty |
| [Avisoft-SASLab Lite](https://avisoft.com/sound-analysis/) | Free, No License | Local  | Windows | Oscillogram, Spectrogram, Power Spectrum | Yes, limited | None | Yes | Yes | Yes, but could not find | Yes| No | No | High difficulty; not intuitive, but has nice documentation online |
| [Avisoft-SASLab Pro](https://avisoft.com/sound-analysis/) |  €2900/$3400 for 1 year License | Local | Windows | Oscillogram, Spectrogram, and Power Spectrum | Yes | Multi, Bounding Boxes | Yes | Yes | Yes, but could not find | Yes | No | No | High difficulty; not intuitive, but has nice documentation online |
| [Kaleidoscope Free](https://www.wildlifeacoustics.com/products/kaleidoscope-pro) | Free, No License | Local | Windows and Linux | Oscillogram, Spectrogram, and Power Spectrum | Yes, limited | Single label for entire file | Yes | No | Yes | No | No | Yes| Easy Difficulty |
| [Kaleidoscope Pro](https://www.wildlifeacoustics.com/products/kaleidoscope-pro) | $399 for 1 year License | Local | Windows and Linux | Oscillogram, Spectrogram, and Power Spectrum | Yes | Single label for entire file | Yes | No | Yes | Yes | No | Yes | Easy Difficulty |
| [Label Studio](https://labelstud.io) | $149/month, additional users $99/month (up to 12 users) | Local, open-source |  Windows and Linux | Oscillogram and Spectrogram (Only goes up to 20kHz) | Yes | Multi, Start-stop | Yes (only to down to 0.5 speed) | No | No | None | Yes, can have as many users as the server can handle | Yes | Easy Difficulty, limited labeling options and spectrogram specifications |
| [PAMGuard](https://www.pamguard.org/) | Free, No License | Local, open-source; has modules that you connect together to create a pipeline | Windows and Linux | Oscillogram and Spectrogram | Yes | Multi, Bounding Boxes | Yes | Yes |  | Yes | No | | Hard Difficulty, have setup module/pipeline before utilizing software |
[Pyrenote-desk](https://github.com/UCSD-E4E/pyrenote-desk) | Free, No License | Local | Linux | Oscillogram and Spectrogram | Yes, limited | Multi, Bounding Boxes, not currently working | Yes | No | No | No | Yes? not currently working | Yes? not currently working | Easy Difficulty|
| [RavenLite](https://www.ravensoundsoftware.com/software/raven-lite/) | Free, with Free License | Local | Windows and Linux | Oscillogram, Linear Spectrogram, Power Spectrum | Yes, limited | Multi, Bounding Boxes | Yes | Yes | Yes | No | No | No batch load/save, Yes batch filtering | Medium Difficulty, lots of buttons and tabs |
| [RavenPro](https://www.ravensoundsoftware.com/software/raven-pro/) | $100 for 1 year License for non-profits | Local | Windows and Linux | Oscillogram, Linear Spectrogram, Power Spectrum | Yes | Multi, Bounding Boxes | Yes | Yes | Yes | Yes | No | No batch load/save, Yes batch filtering | Medium Difficulty, lots of buttons and tabs |

### Legend:
- Annotation Types: 
  - Multi = software can create multiple labels
  - Start-stop = software only creates labels with a start-stop time, no frequency information
  - Bounding Box = software creates bounding box that includes time and frequency information
  - None = software does not have annotation capabilites 

- Sound Measurements = can the software measure call parameters (i.e. min/max/avg frequency, call duration, peak frequency, etc.)? Important for bat species identification!

- ??? = Software claims it can perform this capability, but could not be reproduced with bat data.

- | blank | = Information not known   

## Notes on each software:
### ARBIMON
__(No longer considering for bats; Decided at Whoot Meeting on 1-13-2026)__

  - Good introduction [video](https://www.youtube.com/watch?v=YRSRdilIVHM) from Carly Batist from the K. Lisa Yang Center for Conservation Bioacoustics YouTube Channel
  - __Pros:__
    - Web-based and cloud-enabled (easy access)
    - Multi-user capability and can assign users different roles/access to each project (can download from cloud as well)
    - Can play audio recordings; can handle high sampling rate (up to 384kHz) AudioMoth data
    - Visualizes the Spectrogram and Energy Density Spectrogram of the data [[timestamp: 12:50](https://youtu.be/YRSRdilIVHM&t=770)]
    - Can view audio with sampling rate up to 384kHz 
    - Can annotate spectrograms (creates bounding boxes, which includes start and end times and frequency max and min)
    - Can create a predetermined template of all the labels and select them
    - Options are available to keep data/project private
    - Includes tagging audio file (i.e. issues with the recording, no species of interest detected, etc.)
    - Includes "Species Presence Validation", which is a drop-down yes-no menu indicating where certain species are contained in the entire audio file (separate from the labels); can import a "species list"
    - Apply filters (high pass, low pass, band pass)
    - Can adjust the gain of the audio
    - Has Pattern Matching (Cross-correlation algorithm), Random Forest Models, Audio Event Detection, and Soundscape (Energy Density Spectrogram?)
      - Clustering has a cap on the size of the playlist (n=?, did not specify) that you can run, but should take less than a day to run
      - Can reach to someone at ARBIMON to increase the clustering cap if needed
    - Also includes 10 Regional Classification CNNs [[Timestamp 28:03](https://youtu.be/YRSRdilIVHM&t=1683)]
      - Panama, Edudor, Puerto Rico, Sumatra, Brazil, ...
      - 670 species
      - Can manually validate the outputs
    - Has a public library of labeled spectrogram data for specific calls for 6000+ species (includes, birds, frogs, and bats)
    - Has an "Insites" tab that provides information about hotspots for species richness, number of species vs. hour of day, detection Frequency, and naive occupancy
    - *Has option to exlude certain animals' location data (i.e. endangered) from public access 
  - __Cons:__
    - Web-based and cloud-enabled (have to upload data to the cloud)
    - Proprietary software
    - No Visualization of Oscillogram
    - __Ownership of any data uploaded to ARBIMON is in question and will need to be negotiated with ARBIMON!__
    
    *Email for inquires: contact@rfcx.org

### Audacity 
__(No longer considering for bats; Decided at Whoot Meeting on 1-7-2026)__
  - __Pros:__
    - Linux and Windows compatible
    - Visualizes Oscillogram, Spectrogram, Power Spectrum (just have to select multi-view for the track by right clicking)
    - Can view audio with sampling rate up to 384kHz 
    - Can create multiple labels (.txt file) but only based on oscillogram (provides: start, stop, label_name)
    - GUI is fairly easy to operate
  - __Cons:__ 
    - Cannot adjust spectrogram visualization (except for zoom) or how the spectrogram is calculated
    - Proprietary software 

## Avisoft-SASLab (Lite/Pro)
  - Avisoft-SASLab Pro is a powerful Windows application (compatible to Vista/7/8/10/11) for investigating animal acoustic communication.
  - __Pros:__
    - Has a Lite free software version 
    - Windows compatible
    - Visualizes oscillogram, spectrogram, power spectrum (Lite only lets you use n_fft=256 and cannot change this setting)
    - Can view audio with sampling rate up to 384kHz 
    - Can adjust playback settings to slow and adjust the pitch to hear the bats.
    - Digital filtering available (High, Low, Bandpass, Bandstop, User-defined, band rejection, and reduction filters)
    - Can perform sound parameter measurements
    - Can perform even detection
    - Can label calls manually but only with Pro
    - Has nice documentation
  - __Cons:__
    - Not Linux compatible
    - Proprietary software
    - GUI is not intuitive (especially in Lite version)
    - Cannot adjust contrast of spectrogram, so it is difficult to see bat calls, unless the call is VERY loud
    - Only labeling option is for single point with time duration.
    - Does not have magma or plasma color scheme for viewing spectrogram
    - Annotations are not available in lite version
    - Pricing for the Pro version is €2900, which is ~$3,400 (on 1/22/26)

### Kaleidoscope Lite (Did not have access to Pro)
  - __Pros:__
    - Linux and Windows compatible
    - Designed for bat analysis workflows; Has bat analysis system (has built-in ID option with Pro)
    - Has time expansion for bat data
    - Visualizes oscillogram and spectrogram
    - Can view audio with sampling rate up to 384kHz 
    - Can adjust spectrogram parameters using the GUI 
    - Can zoom-in and out of spectrogram to focus on a particular part of the audio file and can play just that highlighted section.
    - Can label with multiple labels
  - __Cons:__
    - GUI is not intuitive, but options are limited in free version; would need to provide a detailed SOP document
    - Can only have 32 labels available at anytime while using the software
    - Labels the entire audio file and does not give start and stop times
    - Will need pre-processiong into smaller chunks (1-5s)
    - Proprietary software
      
    *Pro version is costly $399 for software per user + Subscription + Cloud storage and data upload costs. Subscription allows two activations for a single user for 12 months.

    **Here is a introduction video for using Kaleidoscope with bat data (https://www.youtube.com/watch?v=eT9XYPpESPw)

### Label Studio (Free version): 
__(No longer considering for bats; Decided at Whoot Meeting on 1-13-2026)__
*Based on the current formating of labelstudio, a team would need to develope and code a labelstudio template or bat version before we can proceed.
  - __Pros:__
    - Web-based software
    - User-friendly GUI
    - Can visualize oscillogram and spectrogram (__only up to 20 kHz__)
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

### PAMGuard
  - Originally made for underwater "click" detections of cetaceans. Can also process bat data and has pipelines for bats from AudioMoth recorders specifically
  - __Pros:__
    - Free Software!
    - Linux and Windows compatible? 
    - Open source program
    - Has been used in published literature for manual annotation of bat data, even from AudioMoth ([Brinkløv et. al, 2022](https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1111/2041-210X.14131))
    - Visualizes oscillogram and spectrogram
    - Can view audio with sampling rate up to 384kHz 
    - Can label calls manually
    - Has a built-in neural network, ANIMAL-SPOT, for event detection and classification of calls
    * Intro to PAMGuard video with bat data: https://www.youtube.com/watch?v=zqwy1daloMY
  - __Cons:__
    - Not intuitive to initially setup, but once module settings are saved, the program will automated load with those modules during future sessions.
    - GUI is not user-friendly; would need to provide a detailed SOP document 

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
    - Adding recordings requires lots of metadata to be inputted first, and I __cannot__ upload recordings without deployment information (which needs to be optional).
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

### RavenPro (RavenLite is similar; it can perform multiple annotations, but cannot adjust spectrogram parameters)
  - __Pros:__
    - Linux and Windows compatible
    - GUI is a little complex, but not difficult to learn; would need to provide a detailed SOP document
    - Visualizes oscillogram, spectrogram, power density spectrum
    - Can view audio with sampling rate up to 384kHz 
    - Can create multiple labels (start-end time or bounding boxes) on the spectrogram (saves as a .txt file, one table file per recording!)
    - Can perform spectral and sound parameter measurements
    - Apply filters (high pass, low pass, band pass, band stop, and more)
    - Can play auto at a slower rate (to "hear" the bats)
    - Can save workspaces, so you can come back to a project
    - Can perform batch processes for band pass filtering
    - Has been used in published literature for manual annotation of bat data, even from AudioMoth ([Brinkløv et. al, 2022](https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1111/2041-210X.14131))
    - Pro version has BirdNet Neural Network built-in for classification of birds, as well as a blue whale, cricket, and narwhal CNNs
  - __Cons:__ 
    - No preset labels, must type in label (could introduce labeling errors from misspelled labels)
    - You can multi-load audio files, but there is no database
    - You cannot save out multiple .txt across multiple sound files, only allows for data from one audio file
    - Can configure the load of an audio file, but the speed/time expansion setting does not seem to work
    - Originally designed for bird calls
    - Proprietary software
    * Pro version costs $100 per license for non-profits, for more information: [RavenPro Pricing](https://www.ravensoundsoftware.com/raven-pricing/)
