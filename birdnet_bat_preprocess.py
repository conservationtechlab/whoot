"""Bat Data Preprocessing Script for BirdNET-Analyzer

    This script can perform pitch shifts (slow), clipping
    audio into specified length clips (clipped), and
    correct and clip labels to properly match new audio.

    See main() for more information.
"""

import os
import glob
import numpy as np
import soundfile as sf
import pandas as pd


def species_dir(audio_dir,
                label_dir,
                species):
    """Short summary of the function.

    More detailed description if needed.

    Args:
        param1 (type): Description of the first parameter.
        param2 (type): Description of the second parameter.

    Returns:
        (type): Description of the return value.

    Raises:
        ExceptionType: Description of when this exception is raised.
    """
    audio_dir_species = audio_dir+species+'/'
    label_dir_species = audio_dir_species + label_dir
    return audio_dir_species, label_dir_species


def file_list(directory,
              file_type=".wav"):
    """Short summary of the function.

    More detailed description if needed.

    Args:
        param1 (type): Description of the first parameter.
        param2 (type): Description of the second parameter.

    Returns:
        (type): Description of the return value.

    Raises:
        ExceptionType: Description of when this exception is raised.
    """
    filenames = glob.glob(directory + '/*'+file_type)
    filenames.sort()
    return filenames


def list_of_filenames(audio_dir_species, audio_file_type,
                      label_dir_species, label_file_type):
    """Short summary of the function.

    More detailed description if needed.

    Args:
        param1 (type): Description of the first parameter.
        param2 (type): Description of the second parameter.

    Returns:
        (type): Description of the return value.

    Raises:
        ExceptionType: Description of when this exception is raised.
    """
    all_audio_files = []
    all_label_files = []
    audio_files = file_list(audio_dir_species, file_type=audio_file_type)
    label_files = file_list(label_dir_species, file_type=label_file_type)
    all_audio_files.extend(audio_files)
    all_label_files.extend(label_files)
    return all_audio_files, all_label_files

def output_path(filename, output_dir, tag=""):
    """Short summary of the function.

    More detailed description if needed.

    Args:
        param1 (type): Description of the first parameter.
        param2 (type): Description of the second parameter.

    Returns:
        (type): Description of the return value.

    Raises:
        ExceptionType: Description of when this exception is raised.
    """

    # Output location:
    no_path_file = filename.split("/")[-1]
    split_filename = no_path_file.split(".", 1)[0]
    file_type = no_path_file.split(".", 1)[-1]
    output_filename = output_dir+split_filename+tag+file_type
    return output_filename


def slow_raven_label(label_df, rate):
    """Short summary of the function.

    More detailed description if needed.

    Args:
        param1 (type): Description of the first parameter.
        param2 (type): Description of the second parameter.

    Returns:
        (type): Description of the return value.

    Raises:
        ExceptionType: Description of when this exception is raised.
    """
    label_start_time = np.array(label_df['Begin Time (s)'])/rate
    label_end_time = np.array(label_df['End Time (s)'])/rate
    label_low_freq = np.array(label_df['Low Freq (Hz)'])*rate
    label_high_freq = np.array(label_df['High Freq (Hz)'])*rate
    label_df["Begin Time (s)"] = label_start_time
    label_df["End Time (s)"] = label_end_time
    label_df["Low Freq (Hz)"] = label_low_freq
    label_df["High Freq (Hz)"] = label_high_freq
    return label_df


def save_clipped_audio(audio,
                       slow_sr,
                       clip_dur,
                       audio_filename,
                       output_dir):
    """Short summary of the function.

    More detailed description if needed.

    Args:
        audio:
        slow_sr:
        clip_dur:
        audio_filename:
        output_dir:

    Returns:
        None: Saves out clipped audio at the slow_sr
    """

    # Calculating audio duration and remainder
    total_len = len(audio)/slow_sr
    clip_total = int(np.ceil(total_len/clip_dur))

    # Clipping audio: (should be turned into a function later...)
    for j in range(0, clip_total):
        if clip_dur*(j+1) > total_len:
            clip_audio = audio[clip_dur*slow_sr*j:]
            zero_padding_dur = (clip_total*clip_dur)-total_len
            zero_padding_clip = np.zeros(int(zero_padding_dur*slow_sr))

            clip_audio = np.concat([clip_audio, zero_padding_clip], axis=0)
        else:
            # Clipping audio to set duration:
            clip_audio = audio[clip_dur*slow_sr*j:clip_dur*slow_sr*(j+1)]

        # Prints output filename:
        split = audio_filename.split('/')[-1]
        print(f'Saving out clip {j+1} out of {clip_total} from {split}')
        # Output location:
        clip_start = str(clip_dur*j)
        clip_end = str(clip_dur*(j+1))
        output_filename = output_path(audio_filename, output_dir,
                                      tag="_"+clip_start+"_to_"+clip_end+"s")
        # Saves slowed audio in the same directory as the input files:
        sf.write(output_filename, clip_audio, slow_sr)


def save_clipped_raven(df,
                       clip_dur,
                       total_len,
                       label_filename,
                       output_dir):
    """Short summary of the function.

    More detailed description if needed.

    Args:
        df
        clip_dur:
        total_len:
        label_filename:
        output_dir:

    Returns:
        None: Saves out clipped label
    """
    # Calculating audio duration and remainder
    # Clipping audio: (should be turned into a function later...)
    for j in range(0, int(np.ceil(total_len/clip_dur))):
        # Use the appropriate delimiter (sep) and other parameters:
        df_new = df[(df['Begin Time (s)'] >= (clip_dur*j)) &
                    (df['End Time (s)'] < (clip_dur*(j+1)))]

        if (df[(df['Begin Time (s)'] <= (clip_dur*(j+1))) &
               (df['End Time (s)'] > clip_dur*(j+1))].empty) and (
                   df[(df['Begin Time (s)'] <= (clip_dur*j)) &
                      (df['End Time (s)'] > (clip_dur*j))].empty):
            # Compute the final dataframe:
            df_final = df_new

        else:
            df_add1 = df[(df['Begin Time (s)'] <= (clip_dur*j)) &
                         (df['End Time (s)'] > (clip_dur*j))]
            df_add2 = df[(df['Begin Time (s)'] <= (clip_dur*(j+1))) &
                         (df['End Time (s)'] > (clip_dur*(j+1)))]
            df_add1.loc[:, "Begin Time (s)"] = clip_dur*j
            df_add2.loc[:, "End Time (s)"] = clip_dur*(j+1)
            # Create new raven selection table based on time
            df_final = pd.concat([df_add1, df_new, df_add2])
            # Adjust time displacement since new audio files starts at 0:
            df_end_time = df_final["End Time (s)"]
            df_begin_time = df_final["Begin Time (s)"]
            df_final.loc[:, "Delta Time (s)"] = df_end_time-df_begin_time
            df_final.loc[:, "Begin Time (s)"] = df_begin_time-(clip_dur*j)
            df_final.loc[:, "End Time (s)"] = df_end_time-(clip_dur*j)
        # Output location:
        clip_start = str(clip_dur*j)
        clip_end = str(clip_dur*(j+1))
        output_filename = output_path(label_filename, output_dir,
                                      tag="_"+clip_start+"_to_"+clip_end+"s")
        # Save out corrected Raven selection table:
        df_final.to_csv(output_filename, sep='\t', index=False)


def main():
    """Main function that runs the entire preprocessing steps.

        Args: *will make a .yaml file in the future...
            species:
            audio_dir: Folder that contains your audio
            label_dir: Folder will be contained in audio_dir 
            audio_output_dir: Slowed audio output folder
            clip_output_dir: Clipped audio output folder
            clip_dur: Desired clip length
            rate: Rate to slow audio file
            audio_file_type: Type of audio file
            label_file_type: Type of label file
            labels_exist: Indicate whether there are labels to edit
            save_clips: Indicate whether to save out clipped data

        Returns:
            (audio file): slowed audio based on defined rate
            (label file): slowed labels based on defined rate (optional)
            (audio files): slowed and clipped audio files (optional)
            (label files): slowed and slipped labels files (optional)

        Raises:
            ValueError: Raises error is the number of annotations and audio files do not match.
    """
    # #######################Specify Parameters################################
    species_id = ['CENCEN', 'MOLMOL', 'SACBIL']
    # ['CENCEN', 'DICALB', 'EPTBRA', 'EPTFUR', 'MOLMOL', 'RHOAEN']
    # Folder that contains your audio:
    audio_dir = '/home/vporter/Desktop/SAN_DIEGO_selection/'
    # Folder will be contained in audio_dir
    label_dir = 'Raven Labels/'
    # Audio output folder
    audio_output_dir = audio_dir+'1. Slowed 10x Xeno-Canto Audio/'
    # Clipped audio output folder
    clip_output_dir = audio_dir+'2. Slowed 3s Clips by Species (BirdNET)/'
    # Desired clip length, in seconds
    clip_dur = 3
    # Rate to slow audio file, Bats: 0.1
    rate = 0.1
    # Type of audio file: default is .wav
    audio_file_type = '.wav'
    # Type of label file: default is raven selection tables
    label_file_type = '.selections.txt'
    # Indicate whether there are labels to edit: Yes == True, No == False
    labels_exist = False
    # Indicate whether to save out clipped data: Yes == True, No == False
    save_clips = True
    # #########################################################################
    # Cycling through each specified species
    for species in species_id:
        # Path string creation for directory of specified species (sp):
        audio_dir_sp, label_dir_sp = species_dir(audio_dir, label_dir, species)
        # Path string creation for all filenames in:
        audio_files_sp, label_files_sp = list_of_filenames(audio_dir_sp, audio_file_type,
                                                           label_dir_sp, label_file_type)
        # Creating output directory/filenames:
        audio_o_dir_sp, label_o_dir_sp = species_dir(audio_output_dir,
                                                     label_dir,
                                                     species)
        audio_clip_o_dir_sp = clip_output_dir+species+"/"
        label_clip_o_dir_sp = clip_output_dir+species+"/"+label_dir
        # Make output audio directory, if needed:
        if not os.path.isdir(audio_o_dir_sp):
            os.makedirs(audio_o_dir_sp)
        # Make output label directory, if needed:
        if not os.path.isdir(label_o_dir_sp):
            os.makedirs(label_o_dir_sp)
        # Make output audio clip directory, if needed:
        if not os.path.isdir(audio_clip_o_dir_sp):
            os.makedirs(audio_clip_o_dir_sp)
        # Make output label clip directory, if needed:
        if not os.path.isdir(label_clip_o_dir_sp):
            os.makedirs(label_clip_o_dir_sp)
        if labels_exist:
            if len(audio_files_sp) != len(label_files_sp):
                raise ValueError("The number of annotations and audio files do not match.")
            for audio_file, label_file in zip(audio_files_sp, label_files_sp):
                # Slowing down the audio by 10x:
                y, sr = sf.read(audio_file)
                slow_sr = int(sr * rate)
                # Saving out slowed audio:
                audio_filename = audio_file.split("/")[-1]
                split_audio_filename = audio_filename.split(".", 1)[0]
                output_audio_filename = output_path(split_audio_filename,
                                                    audio_o_dir_sp,
                                                    tag="_slowed")
                sf.write(output_audio_filename, y, slow_sr)
                # Loading label file:
                label = pd.read_csv(label_file, sep='\t')
                # Slowing down the label times by 10x:
                label_df_slowed = slow_raven_label(label, rate)
                # Saving out slowed labels as individual raven files:
                label_filename = label_file.split("/")[-1]
                split_label_filename = label_filename.split(".", 1)[0]
                output_label_filename = output_path(split_label_filename,
                                                    label_o_dir_sp,
                                                    tag="_slowed")
                label_df_slowed.to_csv(output_label_filename, sep='\t', index=False)

                if save_clips:
                    total_len = len(y)/slow_sr
                    # Creating 3s Clips of Slow Audio and Saving Out:
                    save_clipped_audio(y,
                                       slow_sr,
                                       clip_dur,
                                       audio_filename+"_slowed",
                                       audio_clip_o_dir_sp)
                    # Creating 3s Clip Labels and Saving Out:
                    save_clipped_raven(label_df_slowed,
                                       clip_dur,
                                       total_len,
                                       label_filename+"_slowed",
                                       label_clip_o_dir_sp)
        else:
            # Slowing audio and label files by 10x:
            for audio_file in audio_files_sp:
                # Slowing down the audio by 10x:
                y, sr = sf.read(audio_file)
                slow_sr = int(sr * rate)
                # Saving out slowed audio:
                slow_tag = "_slowed.wav"
                a_filename = audio_file.split("/")[-1]
                len_a_type = len(audio_file_type)
                audio_slowed_filename = a_filename[:-len_a_type]+slow_tag
                output_audio_filename = audio_o_dir_sp+audio_slowed_filename
                sf.write(output_audio_filename, y, slow_sr)
                if save_clips is True:
                    # Creating 3s Clips of Slow Audio and Saving Out:
                    save_clipped_audio(y,
                                       slow_sr,
                                       clip_dur,
                                       audio_slowed_filename,
                                       clip_output_dir+species+"/")


main()
