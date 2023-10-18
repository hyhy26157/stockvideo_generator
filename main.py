from audio_transcript_whisper import audio_transcript_whisper
from get_query import summerisation_general
from download_video import Download_Video
from text_video_combine import *
from transcript_to_dataframe import *
import pandas as pd
import time
import yaml
import asyncio
import os
import subprocess

async def main():

    
    cf.change_settings({"IMAGEMAGICK_BINARY": r"D:\ImageMagick-7.1.0-Q16-HDRI\magick.exe"})
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    #get filename of all current directory
    file_list = os.listdir('.')

    if 'output.json' not in file_list:
        audio_transcript_whisper() #use ai model to transcript the video

    if 'summary.json' not in file_list:
        smart_df = smart_reframe_df(timestamp_to_df(r'output.json'),words_per_row = config['word_per_row'])
        summerisation_general(smart_df,num_char_prompt = config['num_char_prompt'],engine=config['summarisation_engine'])

    download_video = Download_Video()
    #get stockvideo based on the transcript summary (query)
    download_video.download_video() 
    
    sv = Stock_Videos_Framerate_Adjust()
    sv.video_framerate_adjuster()

    # folder_path = '.' 
    # for filename in os.listdir(folder_path):
    #     if filename.endswith('.mp4'):  
    #         file_path = os.path.join(folder_path, filename)  
    #         os.remove(file_path)  
    
    vt = Video_Text_Combine(text_width = config['stockvideo_text_width'], fontsize= config['stockvideo_fontsize'], speech_file = config['speech_file'])
    vt.video_text_combine() #combine video and text
    vt.videos_texts_combine() #concatenate all videos into 1

    folder_path ='adjusted_framerate_video_folder' 
    for filename in os.listdir(folder_path):
        if filename.endswith('.mp4'):  
            file_path = os.path.join(folder_path, filename)  
            os.remove(file_path)
    

if __name__ == '__main__':
    asyncio.run(main())


