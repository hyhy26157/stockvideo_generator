# Import API class from pexels_api package
from Pexels import Client
from Keys.keys import PEXELS_API_KEY
import pandas as pd
from PIL import Image
from io import BytesIO
import time
import sys
import requests
from itertools import combinations
import concurrent.futures
from moviepy.editor import *
from get_query import thesaurus_webster
from get_query import emotion_to_action
from collections import Counter
import random
import yaml
import os

class Download_Video:
    #define the whether to get video from personal collection or keywords 
    def __init__(self):
        self.config = self.load_config()
        self.downloaded_videos_urls = []
        self.num_of_retries = self.config['retries']
        self.video_orientation = self.config['download_video_orientation']
        self.download_type = self.config['download_type']

    def load_config(self):
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)

    #download stock videos from pexel based on conditions
    def download_video(self):

        self.old_stockvideo_json_file_check()

        if os.path.exists(self.config['speech_transcript_json_v1']):

            with open(self.config['speech_transcript_json_v1'], 'r') as f:
                content = f.read()

            #turn json file to dataframe
            df = pd.read_json(content, orient= 'columns')

            #duration starting afresh from the last downloaded index
            duration_baggage = 0 # download longer video for prev unscrapped transcript query

        else:
            #if running the first time.
            with open(self.config['speech_transcript_json_v0'], 'r') as f:
                content = f.read()

            #turn json file to dataframe
            df = pd.read_json(content, orient= 'columns')

            # setting new columns
            df['best_video_url'] = ''
            df['logs'] = ''
            df['video duration'] = 0
            df['duration'] = df['time_end'] - df['time_start']
            df['baggage'] = 0
            df['emotion'] = df['query']
            df['gender'] = self.config['gender']
            duration_baggage = 0 # download longer video for prev unscrapped transcript query

            #make queries all lower case
            df['query'] = df['query'].str.lower()

            # manipulate query column
            df,emotions_one_word_actions_dict = emotion_to_action(df)

        print(f"There are {len(df)} of lines in the transcript")

        file_name_list = self.get_file_name_list()

        print(file_name_list)
        try:
            biggest_file_index = max(list(map(int, file_name_list)))
            #adjust to the first same query to deal with debt
            while True:
                print(biggest_file_index)
                if (biggest_file_index > 0) and (df.loc[biggest_file_index,'query'] == df.loc[biggest_file_index-1,'query']):
                    biggest_file_index -= 1
                    continue
                break
        except ValueError:
            biggest_file_index = 0
        
        # to handle remaining debt where last index has no video downloaded.
        last_video_query = False

        #give additional buffer to eliminate fringe case where video duration is just nice which gave error.
        buffer = 1

        while True:
            
            # reinterate the last df row
            if last_video_query == True:
                last_video_query = False
                last_download_index = df[df['logs'].str.contains('downloaded')].index
                last_video_download_query = df.loc[last_download_index,'query']
                df.loc[-1,'query'] = last_video_download_query
                df = df[-1]

            #get stock video based on query 

            for index, row in df.iterrows():
                print(f'now at index {index}')

                # continue to the last downloaded index file
                if last_video_query == False:
                    if biggest_file_index != 0:
                        biggest_file_index -= 1
                        continue
                

                video_query = df.loc[index,'query']
                
                t_duration = row['duration']
                df.loc[index,'duration'] = t_duration
                slients = row['slient_between_lines']
                df.loc[index,'slient_between_lines'] = slients

                # increase the debt by skipping index to last row with same query keyword
                if (index != df.index[-1]) and (df.loc[index,'query'] ==  df.loc[index+1,'query']):
                    df.loc[index,'best_video_url'] = ''
                    duration_baggage = float(t_duration) + duration_baggage + float(slients)
                    df.loc[index,'logs'] = f'video too short'
                    print(f'skipping to the last same query, baggage is now at {duration_baggage}')
                    df.loc[index,'baggage'] = duration_baggage
                    continue

                print(f'Index {index}: transcript duration is {t_duration + slients} and query is {video_query}')

                # use similar word from the query word to search for more videos
                #using a set of chosen action as alt words
                filtered_alt_word_list = []
                #so that querying would not follow a pattern.
                random.shuffle(emotions_one_word_actions_dict[row['emotion']]) #inplace operation
                alt_word_list = emotions_one_word_actions_dict[row['emotion']][:self.config['no_of_query']]
                
                #look for unique word in query only
                for alt_word in alt_word_list:
                    if alt_word not in df.loc[:,'query']: 
                        filtered_alt_word_list.append(alt_word)
                    else:
                        pass

                # remove query word and only use action words
                filtered_alt_word_list = [video_query] + filtered_alt_word_list
                # add in the gender in the query words
                filtered_alt_word_list = [self.config['gender'] + " " + element for element in filtered_alt_word_list]

                worker_numbers = self.config['concurrent_workers']
                #nested list by number of workers

                def chunk_list(lst, num_chunks):
                    avg = len(lst) / float(num_chunks)
                    last = 0.0
                    chunks = []

                    while last < len(lst):
                        chunks.append(lst[int(last):int(last + avg)])
                        last += avg

                    return chunks

                filtered_alt_word_nestedlist = chunk_list(filtered_alt_word_list, worker_numbers)
                # print("filtered_alt_word_nestedlist",filtered_alt_word_nestedlist)
                

                # create a thread pool with the maximum number of threads
                with concurrent.futures.ThreadPoolExecutor(max_workers=worker_numbers) as executor:
                    # submit a download task for each URL
                    if self.download_type == 'keywords':
                        futures = [executor.submit(self.pexel_video_search, alt_word) for alt_list in filtered_alt_word_nestedlist for alt_word in alt_list]
                    elif self.download_type == 'collection':
                        pass
                    else:
                        print('please choose correct type')
                        sys.exit()
                    print(futures)
                    # wait for`` the download tasks to complete
                    if self.download_type == 'keywords':
                        responses = [future.result() for future in concurrent.futures.as_completed(futures)]
                        multiple_responses = [item for sublist in responses for item in sublist]
                    elif self.download_type == 'collection':
                        #responses = [future.result() for future in concurrent.futures.as_completed(futures)]
                        #print(responses)
                        multiple_responses = self.pexel_video_collection(row['emotion'])
                        print(multiple_responses)


                if len(multiple_responses) > 0:

                    #get video url and duration
                    video_duration_url = []
                    for video in multiple_responses:
                        video_duration_url.append(
                            [
                                video.duration,
                                [video_files.link for video_files in video.video_files],
                                [video_files.quality for video_files in video.video_files],
                                video.image,
                                [video_files.width for video_files in video.video_files],
                                [video_files.height for video_files in video.video_files]
                            ])
                    #print(video_duration_url)
                    
                    # determine brightness for each video
                    # print('getting the brightness of each video...')
                    video_duration_url_image_list = [x[3] for x in video_duration_url]
                    #print(video_duration_url_image)

                    #check brightness of video using image as proxy
                    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_numbers) as executor:
                        futures = [executor.submit(self.process_url_image, video_duration_url_image) for video_duration_url_image in video_duration_url_image_list]
                    # append video brightness in the list to 6th position.
                    responses = [future.result() for future in concurrent.futures.as_completed(futures)]

                    for x,y in zip(video_duration_url,responses):
                        x.append(y)

                    #filter video url, return duration and url only
                    filtered_video_duration_url = []
                    print(f'number of videos found is {len(video_duration_url)}')
                    for video_duration_url_single in video_duration_url:
                        filtered_video_duration_url.append(self.url_filter(video_duration_url_single))
                    video_duration_url = [x for x in filtered_video_duration_url if x is not None]
                    
                    #check video url validity
                    print(f"number of videos passed the filter is {len(video_duration_url)}")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_numbers) as executor:
                        futures = [executor.submit(self.process_url, video_duration_url_single) for video_duration_url_single in video_duration_url]
                    responses = [future.result() for future in concurrent.futures.as_completed(futures)]

                    #remove none in results
                    results = [x for x in responses if x is not None]
                    print("len(results)",len(results))
                    url_result = [x[1] for x in results]

                    #get unique results only. now video_duration_url only has duration and url
                    video_duration_url = [t for t in results if url_result.count(t[1]) == 1]
                    print(f"There are {len(video_duration_url)} video url after filter")

                    print('video_duration_url',video_duration_url)

                    try:
                        longest_video = max(x[0] for x in video_duration_url)
                    except (ValueError,TypeError):
                        longest_video = 0
                        print(f'all video failed the requirement. getting alt words...')
                        continue
                    
                    # sort video by duration so video that meets the requirement can be downloaded.
                    if self.download_type == 'keywords':
                        video_duration_url = sorted(results, key=lambda x: x[0])
                    elif self.download_type == 'collection':
                        video_duration_url = sorted(results, key=lambda x: x[0])
                    else:
                        print('no download type provided')
                        sys.exit()

                    video_durations = [x[0] for x in video_duration_url]
                    print(f"video_duration -- {video_durations}")

                    if longest_video >= (t_duration + duration_baggage + slients + buffer):
                        print(f'longest videos passed the filter and baggage !')
                    elif sum(video_durations) >= (t_duration + duration_baggage + slients + buffer):
                        print(f'all sum videos passed the filter and baggage !')
                    else:
                        print(f'no possible combination of videos can make the minimum video duration...')
                        break
                

                #no url passed the filter, going to next index
                print(f'video_duration_url: {video_duration_url}')
                if len(video_duration_url) == 0:
                    duration_baggage = float(t_duration) + duration_baggage + float(slients)
                    print(f'no good video found, going to next index..., baggage now at {duration_baggage}')
                    df.loc[index,'baggage'] = duration_baggage
                    df.loc[index,'logs'] = f'no good video found'
                    continue

                # Filter duration
                best_video_url = []

                #flag to check if there is any video gotten from the index.
                video_gotten = False
                print(f'checking if any of the video duration is enough to handle baggages')
                
                # use 1 video per index to deal with baggage
                if longest_video > (t_duration + duration_baggage + slients):
                    # Find video duration that passed the video duration requirement

                    try:
                        for v_duration,url in video_duration_url:
                            #print(f'url to get is {url}')
                            print(f'looking for video that passed v_duration requirement..')
                            if (v_duration > (t_duration + duration_baggage + slients + buffer)):  
                                #print(f"got a video : v_duration {v_duration} bigger than baggages {t_duration + duration_baggage + buffer + slients}")
                                print(f"Getting video from with a v_duration of {v_duration} sec...")
                                # instantiate get_video
                                if self.retries(index,url,video_name = (self.config['gender'] + " " + video_query)) == False:
                                    print(f'video cannot download after 3 retries...getting another url..')
                                    continue

                                #log video duration
                                name_of_video = 'index ' + str(index)
                                video_nownow = VideoFileClip(f'{name_of_video}.mp4')
                                df.loc[index,'video duration'] += video_nownow.duration
                                video_nownow.close()

                                duration_baggage = 0
                                df.loc[index,'baggage'] = duration_baggage
                                gender = self.config['gender']
                                print(f'Index {index}: {gender} {video_query} video downloaded from 1 location')
                                video_duration_now = df.loc[index,'video duration']
                                df.loc[index,'logs'] = f'video downloaded from 1 location with duration of {video_duration_now}'
                                best_video_url.append(url)
                                
                                video_gotten = True
                                break
                    

                    # deal with fringe cases where only last set of alt words passed filter (room for improvement area)
                    except ValueError:
                        
                        for v_duration,url in video_duration_url:
                            #print(f'url to get is {url}')
                            print(f'looking for video that passed v_duration requirement..')
                            if (v_duration > (t_duration + duration_baggage + slients + buffer)):  
                                #print(f"got a video : v_duration {v_duration} bigger than baggages {t_duration + duration_baggage + buffer + slients}")
                                print(f"Getting video from with a v_duration of {v_duration} sec...")
                                # instantiate get_video
                                if self.retries(index,url,video_name = f"{row['gender']} + ' ' + {row['query']}") == False:
                                    print(f'video cannot download after 3 retries...getting another url..')
                                    continue
                            

                                #log video duration
                                name_of_video = 'index ' + str(index)
                                video_nownow = VideoFileClip(f'{name_of_video}.mp4')
                                df.loc[index,'video duration'] += video_nownow.duration
                                video_nownow.close()

                                duration_baggage = 0
                                df.loc[index,'baggage'] = duration_baggage
                                print(f'Index {index}: {video_query} video downloaded from 1 location')
                                video_duration_now = df.loc[index,'video duration']
                                df.loc[index,'logs'] = f'video downloaded from 1 location with duration of {video_duration_now}'
                                best_video_url.append(url)
                    
                                
                                video_gotten = True
                                break
                        

                    #if single video didn't meet requirement
                    if video_gotten == False:
                        duration_baggage = float(t_duration) + duration_baggage + float(slients)
                        df.loc[index,'baggage'] = duration_baggage
                        video_duration_now = df.loc[index,'video duration']
                        df.loc[index,'logs'] = f'no good video found'
                        print(f'no good video found, baggage at {duration_baggage}')
                        use_multiple_video_flag = True
                        # use multiple video to handle the video
                        pass
                            

                    # store it in the dataframe
                    df.loc[index,'best_video_url'] = str(best_video_url)

          

                # use >1 video per index to deal with baggage
                elif (longest_video < (t_duration + duration_baggage)) or use_multiple_video_flag == True:

                    use_multiple_video_flag = False #handling video_gotten flag == false situation
                    print(f'longest video cannot handle the baggages')
                    print(f'using >1 videos instead...')
                    print(f'baggage is now at {duration_baggage}')
                    df.loc[index,'baggage'] = duration_baggage
                    sorted_list = sorted(video_duration_url, key=lambda x: x[0], reverse=True)
                    print(f'sorted list are {sorted_list}')
                    list_durations = [x[0] for x in sorted_list]
                    print(f'all the durations are {list_durations}')

                    #get the best possible video combination based on video duration
                    target = t_duration + duration_baggage + slients
                    print(f'video duration to get should be {target}')
                    best_videos_url_duration = self.find_closest_sum(list_durations, target)
                    if (best_videos_url_duration == False):
                        duration_baggage = float(t_duration) + duration_baggage + float(slients)
                        df.loc[index,'baggage'] = duration_baggage
                        print(f'no possible combination of videos to handle the baggages')
                        print('assuming too little videos that meet the requirement to handle the baggage')
                        print(f'going to next index, baggage now at {duration_baggage}')
                        
                        continue

                    print(f'to download {len(best_videos_url_duration)} videos to handle the baggages')
                    print(f'duration are {best_videos_url_duration}')

                    # instantiate get_video
                    name_counter = 0

                    best_videos_url_duration = list(best_videos_url_duration)
                    while len(best_videos_url_duration) != 0:
                        return_index = list_durations.index(best_videos_url_duration[0])
                        name_counter += 1
                        #to get video with different url when download videos duration has the same duration.
                        #this only works for sorted duration list
                        count_same_duration = best_videos_url_duration.count(best_videos_url_duration[0])
                        print(f'this video duration count is {count_same_duration}')

                        for i in range(count_same_duration): # i give minimum 0
                            print(f'i value is {i}')
                            print(f"Getting video from {sorted_list[return_index+i][1]}...")
                            print(f"this video has a duration of {list_durations}")
                            if self.retries(index,sorted_list[return_index+i][1],video_name = f"{row['gender']} + ' ' + {row['query']}") == False:
                                print(f'video cannot download after 3 retries..trying to get other url..')

                                #no videos to handle the baggage. moving on to next query.
                                break

                        #log video duration
                        try:
                            name_of_video = 'index ' + str(index)
                            video_nownow = VideoFileClip(f'{name_of_video}.mp4')
                        except OSError:
                            name_of_video_variant = 'index ' + str(index) + '_' + str(name_counter)
                            video_nownow = VideoFileClip(f'{name_of_video_variant}.mp4')

                        df.loc[index,'video duration'] += video_nownow.duration
                        video_nownow.close()

                        #deal with duplicate video names
                        name_of_video_variant = 'index ' + str(index) + '_' + str(name_counter)
                        os.rename(name_of_video +'.mp4', name_of_video_variant +'.mp4' )
                        video_duration_now = df.loc[index,'video duration']
                        best_video_url.append(sorted_list[return_index][1])

                        best_videos_url_duration.pop(0)

                    print(f'video downloaded from multiple location with total duration of {video_duration_now}!')
                    df.loc[index,'logs'] = f'video downloaded from multiple location'
                    df.loc[index,'best_video_url'] = str(best_video_url)
                    
                    duration_baggage = 0

             


                #if no video found for the main loop index
                else:
                    if index == df.index[-1]:
                        last_video_query = True
                    # store it in the dataframe
                    df.loc[index,'best_video_url'] = ''
                    duration_baggage = float(t_duration) + duration_baggage + float(slients)
                    logs = f'no video available'
                    print(f'no video available, baggage is now at {duration_baggage}')
                    
                    
                #storing it in summary.json file
                with open(self.config['speech_transcript_json_v1'], 'w') as out:
                    out.writelines(df.to_json(orient = 'columns'))

            if last_video_query == False:
                break
        
            if self.url_in_dataframe_check(df) == False:
                print(f'videos found on dataframe index is not the same as local drive video index!')

    def old_stockvideo_json_file_check(self):
        file_paths = [self.config['speech_transcript_json_v1'], self.config['speech_transcript_json_v2']]

        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"The file '{file_path}' has been deleted.")
            else:
                print(f"The file '{file_path}' does not exist.")

    #get video from url with retries
    def retries(self,index:int = -1,video_url:str = 'https://www.pexels.com/video/flickers-of-light-in-the-air-at-night-2297636/',video_name:str = 'no_video_name'):
        

        if video_url not in self.downloaded_videos_urls:
            self.downloaded_videos_urls.append(video_url)
        else:
            print(f'video already downloaded. looking for other video')
            return False

        video_name_generator = 'index '+str(index)
        response = requests.get(video_url, stream=True)

        # Check if the response was successful
        if 200 <= response.status_code < 300:
            # Open a local file and write the response content to it in chunks
            with open(f'{video_name_generator}.mp4', 'wb') as file:
                for chunk in response.iter_content(chunk_size=4048576):
                    if chunk:
                        file.write(chunk)
            print(f'Video {video_name} downloaded successfully!')
            return True

        else:
            time.sleep(1)
            if self.num_of_retries != 0:
                self.num_of_retries -= 1
                print(f'video download failed. {self.num_of_retries+1} retries left')
                self.retries(index=index, video_url=video_url, video_name = video_name)
            else:
                print(f'video download stopped')
                return False

    #check if url in dataframe correctly
    def url_in_dataframe_check(self,df:object):
        index_of_all_url = df.index[df['best_video_url'] != '']
        file_names = self.get_file_name_list()
        if Counter(index_of_all_url) == Counter(file_names):
            return True
        else:
            return False

    #get all the video file name in the local
    def get_file_name_list(self):
        
        # Get a list of all files name in the current directory
        files = os.listdir(r"./")
        file_name_list = []
        for file_name in files:
            # Split the vidoe file name and extension
            if '.' in file_name:
                name = file_name.split('.')[0]
                if 'index' in name:
                    index = name.split(' ')[1]
                    if '_' in index:
                        index = index.split('_')[0]
                        file_name_list.append(index)
                    else:
                        file_name_list.append(index)
            else:
                pass

        return file_name_list

    #find the best videos to deal with baggage
    def find_closest_sum(self, values, target):
        # Generate all possible combinations of values from the list
        all_combinations = []
        print(values)
        number_of_values = len(values)
        # start with searching for 2 combo videos, if no combo, search for +1 more combo...
        r = 2
        while True:
            all_combinations.extend(list(combinations(values, r)))
        
            # Find the combination with the smallest absolute difference from the target
            print('before ', all_combinations)
            all_combinations = [v for v in all_combinations if sum(v) > target]

            if len(all_combinations) == 0:
                r += 1
                if (r == number_of_values) or r > 30:
                    break
                continue
            break
        print('after ',all_combinations)
        try:
            closest_combination = min(all_combinations)
            return closest_combination
        except ValueError:
            print(f'no good combination...')
            return False

    def url_filter(self,url):
        #print(url)
        if url[6] >= self.config['download_video_brightness']: #scalar
            #print('brighness failed!')
            return None

        if self.config['download_video_video_quality'] in url[2]: #list
            video_quality_idx = url[2].index(self.config['download_video_video_quality'])
        else:
            #print('video quality failed!')
            return None

        if url[4][video_quality_idx] ==  self.config['download_video_width'] and url[5][video_quality_idx] ==  self.config['download_video_height']: #list
            #print('video height or weight failed!')
            return url[0],url[1][video_quality_idx] #return duration and video url

        else:
            return None

    #check video url validity
    def process_url(self, elem):
        #sort by strictest filter
        response = requests.get(elem[1])
        if response.status_code >= 200 or response.status_code < 300:
            #return duration and url
            return elem[0],elem[1]
            
    #check image url validity
    def process_url_image(self,url):
        response = requests.get(url)
        if response.status_code >= 200 or response.status_code < 300:
            img = Image.open(BytesIO(response.content))
            gray_img = img.convert('L')
            brightness = sum(gray_img.getdata()) / (gray_img.size[0] * gray_img.size[1])
            return brightness
        else:
            return 999

    def save_logs(self):
        # Open a file for writing the logs
        with open('download_video_logs.txt', 'a') as f:
            # Redirect the standard output to the file
            sys.stdout = f

            # Your program code here

        # Reset the standard output
        sys.stdout = sys.__stdout__

    def pexel_video_search(self, video_query:str):

        pexel = Client(token=PEXELS_API_KEY)

        while True:

            try:
                search_videos = pexel.search_videos(
                    query=video_query,
                    orientation=self.video_orientation, 
                    size=self.config['download_video_size'], 
                    locale='', 
                    page=1, 
                    per_page=self.config['download_video_video_per_page']
                    )
                #print(search_videos)
                return search_videos.videos
            except (requests.exceptions.JSONDecodeError,KeyError):
                continue

    def pexel_video_collection(self, emotion):

        pexel = Client(token=PEXELS_API_KEY)

        emotion_collection_id = {'sadness':'7ib4tuu','surprise':'oellmme','joy':'incoyf9','anger':'tcnuo8v','fear':'vfym9t7'}

        media = pexel.get_collection_media(id = emotion_collection_id[emotion])
        
        all_media_video_class = [x for x in media.media]
        return all_media_video_class
