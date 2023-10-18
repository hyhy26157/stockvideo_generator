import pandas as pd
import json
from moviepy.editor import *
import moviepy.config as cf
import cv2
import os
import textwrap
import re
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.audio.AudioClip import AudioClip
import moviepy.video.fx.all as vfx
import shutil
import json
import sys
import pandas as pd
import yaml



class Video_Text_Configuration:

    def __init__(self):
        self.config = self.load_config()
        self.videotext_clip_folder = self.config['final_clip_folder']
        self.stockvideo_df_final = r'./summary_stockvideotext.json' or None
        self.stockvideo_df_inital = r'./summary_stockvideo.json' or None
        self.stock_video_framerate_adjusted_file_location = self.config['adjusted_frame_clip_folder']
        self.stock_videos_combined_file_location = self.config['final_clip_folder']
        self.df = self.__get_clean_stockvideo_df()
        self.stock_videos_combined_filenames = os.listdir(self.config['final_clip_folder'])
        self.all_stockvideos_filenames = os.listdir(self.config['adjusted_frame_clip_folder'])
        self.all_widths_min, self.all_heights_min = self.__set_stockvideos_min_height_width()

    def load_config(self):
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)

    def _get_filenames(self, file_location, only_index = False):
        file_names = os.listdir(file_location)
        
        if only_index == True:
            filename_indexes = []
            for file in file_names:
                file = file.split('.')[0]
                file = int(file.split(' ')[1])
                filename_indexes.append(file)
            return filename_indexes

        return file_names

    def __get_clean_stockvideo_df(self):
        
        def get_stockvideo_df():
            
            #continue where is left off
            try:
                with open(self.stockvideo_df_final, 'r') as f:
                    content = f.read() 

            except FileNotFoundError:
                try:
                    #if running the first time.
                    with open(self.stockvideo_df_inital, 'r') as f:
                        content = f.read()
            
                except FileNotFoundError: #no stockvideo 
                    return pd.DataFrame()

            #turn json file to dataframe
            df = pd.read_json(content, orient= 'columns')

            return df

        df = get_stockvideo_df()
        print(df)
        if df.empty:
            return pd.DataFrame()

        #creating new column
        df['trimmed_video_duration'] = 0

        return df

    def __set_stockvideos_min_height_width(self):
        
        all_widths = []
        all_heights = []
        print(f"self.all_stockvideos_filenames {self.all_stockvideos_filenames}")
        print(f'getting the min width and height from all videos downloaded')
        for video in self.all_stockvideos_filenames:
            video = VideoFileClip(f"{self.stock_video_framerate_adjusted_file_location}{video}").volumex(0)
            all_widths.append(video.w)
            all_heights.append(video.h)
        print(all_widths,all_heights)
        return min(all_widths),min(all_heights)

class Video_Text_Combine(Video_Text_Configuration):
    
    def __init__(self,text_width, fontsize, speech_file):
        super().__init__()
        self.text_width = text_width
        self.fontsize = fontsize
        self.audio_clip = AudioFileClip(speech_file)
        self.df_dict = {key: [] for key in list((self.df.columns))}
        self.new_df = pd.DataFrame.from_dict(self.df_dict)
        self.df_rows = self.df.shape[0]
        #self.current_index = self.__check_last_created_videotext_clip()
        self.current_index = self.__check_last_created_videotext_clip()
        self.text = self.__text_warp(self.df.loc[self.current_index,'transcript'])
        self.query = self.df.loc[self.current_index,'query']
        self.duration_to_trim = self.df.loc[self.current_index,'duration']
        self.logs = self.df.loc[self.current_index,'logs']
        self.slients = self.df.loc[self.current_index,'slient_between_lines']
        self.logs = self.df.loc[self.current_index,'logs']
        self.emotion = self.df.loc[self.current_index,'emotion']
        self.remaining_clip = ''
        self.debt = {'text':[],
                    'duration_to_trim':[],
                    'index':[],
                    'query':[],
                    'slients':[]}

    def videos_texts_combine(self):

        #create a dict for all the sorted files index
        sorted_stockvideos = []

        #get partition text video object back initial downloaded file object to avoid memory issue.
        stockvideos_indexes = self.get_videotext_clip_filenames_indexes()
        sorted_stockvideos_indexes = sorted(stockvideos_indexes)
        print(f'stockvideos_indexes {stockvideos_indexes}')

        for index_idx in sorted_stockvideos_indexes:
            stockvideo = VideoFileClip(f"{self.stock_videos_combined_file_location}index {index_idx}.mp4")
            sorted_stockvideos.append(stockvideo)
        print(f'video flie stored!')
            
        #self.__check_fps_duration(sorted_stockvideos)
        final_clip = concatenate_videoclips(sorted_stockvideos).set_audio(self.audio_clip)
        
        #fading effect for the video
        fcwa_v2 = vfx.fadein(final_clip, self.config['video_fadein'], initial_color=self.config['video_fadein_color'])
        fcwa_v2 = vfx.fadeout(fcwa_v2, self.config['video_fadeout'], final_color=self.config['video_fadeout_color'])
        fcwa_v2.write_videofile(f'{self.stock_videos_combined_file_location}combined_video.mp4',fps = self.config['stockvideo_framerate_set'], threads = self.config['video_writing_threads_number'],codec="libx264")

        #delete remaining video inside adjusted frame folder
        # for filename in os.listdir(self.stock_video_framerate_adjusted_file_location):
        #     if filename.endswith('.mp4'):  
        #         file_path = os.path.join(self.stock_video_framerate_adjusted_file_location, filename)  
        #         os.remove(file_path)

        print(f'file combined with audio!')

    def video_text_combine(self):

        while self.current_index < self.df_rows - 1:
            self.__video_debt_handler()
        
        print(self.df)
        with open('summary_stockvideotext.json', 'w') as out:
            out.write(self.df.to_json())

    def __video_debt_handler(self):
    
        print(f"current index in {self.current_index}")

        if self.__check_video_log(self.logs) == True:
            self.__debt_collector()
            self.__inner_video_debt_handler()
            print(f"current index is set to {self.current_index}")

        #fringe case where 0 index log have video
        else:
            self.remaining_clip = VideoFileClip(f"{self.stock_video_framerate_adjusted_file_location}{videofilename}.mp4").without_audio().resize((self.all_widths_min,self.all_heights_min))
            videofilename = self.__get_filename_handler('stockvideo', self.current_index)
            final_clip = self.__get_textvideo(self.text,self.duration_to_trim,self.current_index,self.query,self.slients)
            final_clip.write_videofile(f'{self.videotext_clip_folder}{videofilename}_v2.mp4',fps = self.config['stockvideo_framerate_set'], threads = self.config['video_writing_threads_number'],codec=self.config['video_codec'])

    def __inner_video_debt_handler(self):

        """
        debt video are stored as object and be consolidated.
        
        
        """

        print(f'Looking for video in the next index at {self.current_index}')
        next_videofilename = self.__get_filename_handler('stockvideo', self.current_index)
        
        while next_videofilename not in self.all_stockvideos_filenames:
            self.__debt_collector()
            next_videofilename = self.__get_filename_handler('stockvideo', self.current_index)
            if self.current_index == self.df_rows -1:
                break

        self.__debt_collector()
            
        print(f"{next_videofilename} found!, reading video...")
        self.remaining_clip = VideoFileClip(f'{self.stock_video_framerate_adjusted_file_location}{next_videofilename}').without_audio().resize((self.all_widths_min,self.all_heights_min))

        # use the found video to deal with debt
        print(self.debt)
        final_clips = []
        debt_length = len(self.debt['index'])
        print("debt_length", debt_length)
        for idx,values in enumerate(zip(*self.debt.values())):
            print("idx values",idx,values)
            if idx >= debt_length-1: #use trailer video in the last debt loop
                final_clip = self.__get_textvideo(*values)
                debt_videofilename = self.__get_filename_handler(self.config['Video_Text_Combine_video_type'],self.debt['index'][idx])
                final_clips.append(final_clip)
            else:
                final_clip = self.__get_textvideo(*values)
                #debt_videofilename = self.__get_filename_handler('stockvideo',self.debt['index'][idx])
                final_clips.append(final_clip)

        
        concatenated_videoclips = concatenate_videoclips(final_clips)
        concatenated_videoclips.write_videofile(f'{self.videotext_clip_folder}{debt_videofilename}', threads=self.config['video_writing_threads_number'],codec=self.config['video_codec'])
        self.__clear_debt()
        
    def __debt_collector(self):
        self.debt['text'].append(self.text)
        self.debt['duration_to_trim'].append(self.duration_to_trim)
        self.debt['index'].append(self.current_index)
        self.debt['query'].append(self.query)
        self.debt['slients'].append(self.slients)
        print(f'debt collected for index {self.current_index}')
        
        #check if current index is reaching last df row.
        if self.current_index == self.df_rows-1:
            pass
        else:
            self.__update_current_info()
            print(f'info updated. index now at {self.current_index}')

    def __get_textvideo(self,text,duration_to_trim,index,query,slients):

        wraped_text = self.__text_warp(text)
        text_clip = self.__text_customisation(wraped_text)
        remaining_clip = self.remaining_clip

        final_clip,remaining_clip = self._set_textvideo(wraped_text,
                                                                    duration_to_trim,
                                                                    index,query,
                                                                    slients,
                                                                    text_clip,
                                                                    remaining_clip)
        self.remaining_clip = remaining_clip

        self.df.loc[index,'trimmed_video_duration'] = final_clip.duration

        return final_clip
    
    def filename_handler(self, index):

        videofilename = "index " + str(index) + ".mp4"

        return videofilename

    def __get_filename_handler(self, video_type, index):

        if video_type == 'stockvideo':
            return self.filename_handler(index)
        else:
            return None

    def __update_current_info(self):
        self.current_index += 1
        self.df['index'] = self.current_index
        self.text = self.__text_warp(self.df.loc[self.current_index,'transcript'])
        self.query = self.df.loc[self.current_index,'query']
        self.duration_to_trim = self.df.loc[self.current_index,'duration']
        self.logs = self.df.loc[self.current_index,'logs']
        self.slients = self.df.loc[self.current_index,'slient_between_lines']

    def __return_debt_clips(self, start_duration, end_duration, remaining_clip):
        video_clip = remaining_clip.subclip(start_duration,end_duration)
        print(f"video clip has a duration of {video_clip.duration}")
        new_start_duration = video_clip.duration
        remaining_clip = remaining_clip.subclip(new_start_duration)
        print(f"remaining_clip has a duration of  {remaining_clip.duration}")
        return video_clip,remaining_clip

    def _set_textvideo(self,text,duration_to_trim,index,query,slients,text_clip,remaining_clip):

        if index == 0:
            initial_slients = self.df.loc[0,"time_start"]
            print(f'duration of the clip is {duration_to_trim} {slients} {initial_slients}')
            clipped_video_clip,remaining_clip = self.__return_debt_clips(0, (duration_to_trim + slients + initial_slients), remaining_clip)
        else:
            clipped_video_clip,remaining_clip = self.__return_debt_clips(0, (duration_to_trim + slients), remaining_clip)
        clipped_video_clip = clipped_video_clip.fx(vfx.colorx, self.config['video_black_overlay'])
        final_clip = CompositeVideoClip([clipped_video_clip, text_clip.set_duration(clipped_video_clip.duration)])        

        return final_clip,remaining_clip

    def __text_warp(self,text) -> str:
        text = textwrap.wrap(text, width=self.text_width)
        text = "\n".join(text)
        return text

    def __check_video_log(self,logs):
        if (('too short' in logs) or ('no good' in logs) or (logs == "")):
            return True

    def __text_customisation(self, text,color='yellow', stroke_color='black', stroke_width=1) -> object:
        text_clip = TextClip(text, fontsize= self.fontsize, color=color, stroke_color=stroke_color, stroke_width=stroke_width)
        text_clip = text_clip.set_position(('center', 'center'))
        return text_clip

    def __check_last_created_videotext_clip(self):
        
        videotext_clip_filenames_indexes = self.get_videotext_clip_filenames_indexes()

        print('getting last created videotext_clip indexes...')
        try:
            return max(videotext_clip_filenames_indexes)
        except ValueError:
            return 0

    def get_videotext_clip_filenames_indexes(self):

        #biggest_final_clip_files_index adjusted to ensure debt handled.
        # Get a list of all files name in the current directory
        videotext_clip_filenames_indexes = []
        for file_name in self.stock_videos_combined_filenames:
            # Split the vidoe file name and extension
            print(file_name)
            if '.' in file_name:
                name = file_name.split('.')[0]
                name = name.split(' ')[1]
                #name = name.split('_')[0]
                videotext_clip_filenames_indexes.append(int(name))
            else:
                pass

        return videotext_clip_filenames_indexes

    # handler list of video object to ensure diff fps doesn't affect the duration when concatenate
    def __check_fps_duration(self, sorted_file_list_values:list): 

        full_video_fps_list = []
        for full_video_file in sorted_file_list_values:
            fps = full_video_file.fps
            full_video_fps_list.append(fps)
        print(f"ull_video_fps_list: ",full_video_fps_list)

        if full_video_fps_list.count(full_video_fps_list[0]) == len(full_video_fps_list):
            return f"all stockvideos' fps are the same"
        
        else:
            print(f"all stockvideos' fps are not the same")
            sys.exit()

    def __clear_debt(self):
        self.debt = {'text':[],
            'duration_to_trim':[],
            'index':[],
            'query':[],
            'slients':[]}
        print(f"debt cleared!")

    def combine_stockvideo_variant(self):

        #open summary
        with open(r"summary_stockvideo.json", 'r') as f:
            content = f.read()

        #turn json file to dataframe
        df = pd.read_json(content, orient= 'columns')

        # Get a list of all files name in the current directory
        file_names = os.listdir(self.stock_video_framerate_adjusted_file_location)
        file_name_list = []
        for file_name in file_names:
            # Split the vidoe file name and extension
            print(file_name)
            if '.' in file_name:
                name = file_name.split('.')[0]
                if 'index' in name:
                    file_name_list.append(name)
            else:
                pass

        #find minimum height and width for all video for resizing
        all_widths = []
        all_heights = []
        print(f'getting the min width and height from all videos downloaded')
        for video in file_name_list:
            video = VideoFileClip(f"{self.stock_video_framerate_adjusted_file_location}{video}.mp4").volumex(0)
            all_widths.append(video.w)
            all_heights.append(video.h)
        print(all_widths,all_heights)
        all_widths_min = min(all_widths)
        all_heights_min = min(all_heights)
        print(f'min width is {all_widths_min} and min height is {all_heights_min}')

        #combining video files with same name, different variant
        videofile_variants = []
        print(file_name_list)
        for x in file_name_list:
            if re.search(r'_\d+', x):
                videofile_variants.append(x)
            else:
                pass

        if videofile_variants:
            print("The list contains file_variants:", videofile_variants)
            print("assuming file_variants are video files, combining them now..")
            same_name_variant = {}
            for file in videofile_variants:
                same_name_variant.setdefault(str(file[:-2]), []).append(file+'.mp4')

            for key in same_name_variant:
                print(key,same_name_variant[key])

                #turn video file into video object file for processing
                temp = []
                for value in same_name_variant[key]:
                    print(value)
                    temp.append(VideoFileClip(f"{self.stock_video_framerate_adjusted_file_location}{value}").volumex(0).resize((all_widths_min,all_heights_min)))
                    
                    #move variant files to a to-be-deleted folder
                    shutil.move(f"./{value}", f"./files_to_delete/{value}")

                print(temp)
                    
                final_clip = concatenate_videoclips(temp)
                print(f'the key is {key}')
                final_clip.write_videofile(f'./{key}.mp4', codec="libx264" , threads=8)
                print(f'file combined!')

            
        else:
            print("The list does not contain duplicates")

class Stock_Videos_Framerate_Adjust():

    def __init__(self):
        self.config = self.load_config()
        self.stock_video_framerate_adjusted_file_location = self.check_stock_video_framerate_adjusted_file_location()
        self.stock_videos_file_location = self.config['stock_videos_folder']
        self.stockvideo_df_final = r'./summary_stockvideotext.json' or None
        self.stockvideo_df_inital = r'./summary_stockvideo.json' or None
        self.df = self.__get_clean_stockvideo_df()

    def load_config(self):
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)

    def check_stock_video_framerate_adjusted_file_location(self):
        adjusted_frame_directory = self.config['adjusted_frame_clip_folder']
        if not os.path.exists(adjusted_frame_directory):
            os.makedirs(adjusted_frame_directory)
        return adjusted_frame_directory

    def __get_clean_stockvideo_df(self):
        
        def get_stockvideo_df():
            
            #continue where is left off
            try:
                with open(self.stockvideo_df_final, 'r') as f:
                    content = f.read() 

            except FileNotFoundError:
                try:
                    #if running the first time.
                    with open(self.stockvideo_df_inital, 'r') as f:
                        content = f.read()
            
                except FileNotFoundError: #no stockvideo 
                    return pd.DataFrame()

            #turn json file to dataframe
            df = pd.read_json(content, orient= 'columns')

            return df

        df = get_stockvideo_df()
        print(df)
        if df.empty:
            return pd.DataFrame()

        #creating new column
        df['trimmed_video_duration'] = 0

        return df

    def filename_handler(self, index):

        videofilename = "index " + str(index) + ".mp4"

        return videofilename

    # def find_stock_videos(self):
    #     file_names = os.listdir(self.stock_videos_file_location)
    #     #get all the stock video name
    #     mp4_files = [file for file in file_names if file.endswith('.mp4')]
    #     return mp4_files

    # check if existing adjusted framrate video folder already completed the job
    def check_existing_stock_video_file(self):
        # List all files in the current directory
        files = os.listdir(r'./')

        # Filter files that end with .mp4
        mp4_files = [file for file in files if file.endswith('.mp4')]
        print(f"mp4 files are {mp4_files}")
        return mp4_files

    def video_framerate_adjuster(self):
        """
        To adjust the framerate of all downloaded video to the set framerate 
        so that it can be concatenate together smoothly.
        
        """
        mp4_files = self.check_existing_stock_video_file()
        
        
        #change the framerate of all stock video one by 1. assume stock video in root folder
        for file_name in mp4_files:
            print(f'dealing with {file_name}')
            #create folder to put frame

            temp_frame_folder = r'./frame_folder/'

            if not os.path.exists(temp_frame_folder):
                    os.makedirs(temp_frame_folder)
            
            vidcap = cv2.VideoCapture(f'{self.stock_videos_file_location}{file_name}')
            assert vidcap.isOpened()

            fps_in = vidcap.get(cv2.CAP_PROP_FPS)
            fps_out = self.config['stockvideo_framerate_set']

            index_in = -1
            index_out = -1
            while True:
                index_in += 1
                out_due = int(index_in * fps_out / fps_in)

                if out_due > index_out:
                    vidcap.set(cv2.CAP_PROP_POS_FRAMES, index_in)
                    success, frame = vidcap.read()

                    if not success: 
                        break

                    index_out += 1
                    cv2.imwrite(f"{temp_frame_folder}{index_out}.png", frame)

                    # do something with `frame`


            # Get the list of frame file names
            frame_files = [f for f in os.listdir(temp_frame_folder) if f.endswith('.png')]

            # Sort the file names in ascending order
            frame_files.sort(key=lambda x: int(x.split(".")[0]))

            # Set the duration of each frame (in seconds)
            frame_duration = 1.0 / fps_out

            clips = [ImageClip(os.path.join(temp_frame_folder, file)).set_duration(frame_duration) for file in frame_files]
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(f'{self.stock_video_framerate_adjusted_file_location}{file_name}', fps=fps_out, bitrate="5000k")


            # Initialize the video clip with the first frame
            # clip = ImageClip(os.path.join(temp_frame_folder, frame_files[0])).set_duration(frame_duration)
            
            # # Loop through the remaining frames and add them to the clip
            # for file in frame_files[1:]:
            #     # Load the frame image and add it to the clip
            #     frame = ImageClip(os.path.join(temp_frame_folder, file)).set_duration(frame_duration)
            #     clip = concatenate_videoclips([clip, frame])

            # # Write the video clip to a file
            # clip.write_videofile(f'{self.stock_video_framerate_adjusted_file_location}{file_name}', fps=fps_out, bitrate="5000k")

            #clean up temp frame folder and files within
            if os.path.exists(temp_frame_folder):
                shutil.rmtree(temp_frame_folder)

