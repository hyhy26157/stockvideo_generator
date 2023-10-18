import json
import pandas as pd
from nltk.corpus import stopwords
import re
import os

emotions_one_word_actions = {
'sadness': ['crying', 'mourning', 'weeping', 'brooding', 'sulking', 'moping', 'lamenting', 'grieving', 'sorrowing', 'pining', 'melancholy', 'dismal', 'downhearted', 'dejected', 'blue', 'woeful', 'gloomy', 'sullen', 'despondent', 'depressed', 'tearful', 'heartbroken', 'tragic', 'devastated', 'bereaved', 'wistful', 'melancholic', 'despairing', 'hopeless', 'disconsolate', 'lonely', 'abandoned', 'neglected', 'rejected', 'betrayed', 'unhappy', 'miserable', 'desolate', 'forlorn', 'disappointed', 'disheartened'],
    'joy': ['smiling', 'laughing', 'dancing', 'singing', 'celebrating', 'rejoicing', 'delighted', 'pleased', 'thrilled', 'excited', 'elated', 'ecstatic', 'euphoric', 'gleeful', 'happy', 'joyful', 'merry', 'jubilant', 'overjoyed', 'blissful', 'contented', 'grateful', 'thankful', 'satisfied', 'fulfilled', 'tranquil', 'peaceful', 'reassured', 'encouraged', 'motivated', 'confident', 'optimistic', 'hopeful', 'enthusiastic', 'passionate', 'energized', 'vibrant', 'lively', 'dynamic', 'radiant','spread cheer', 'dance freely', 'sing joyfully', 'laugh heartily', 'smile brightly', 'rejoice always', 'embrace warmly', 'radiate happiness', 'burst with joy', 'share laughter', 'feel alive', 'experience bliss', 'jump for joy', 'live fully', 'enjoy moments', 'bask in joy', 'shine brightly', 'glow with happiness', 'delight in life', 'be happy'],
    'anger': ['angry','yell', 'shout', 'scream', 'fume', 'rage', 'fury', 'outrage', 'livid', 'incensed', 'mad', 'annoyed', 'irritated', 'frustrated', 'exasperated', 'clench fists', 'grind teeth', 'hit things', 'kick things', 'throw things','slam things', 'slamming doors', 'throwing things','scold', 'passive aggressive','aggressive'],
    'fear': ['trembling', 'shivering', 'cowering', 'flinching', 'fleeing', 'hiding', 'escaping', 'avoiding', 'dreading', 'fearing', 'panicking', 'terrifying', 'startling', 'frightening', 'scaring', 'horrifying', 'intimidating', 'menacing', 'petrifying', 'overwhelmed', 'paralyzed', 'anxious', 'nervous', 'tense', 'uneasy', 'worried', 'scared', 'panicky', 'horrified', 'paranoid', 'jumpy', 'frozen', 'trapped', 'vulnerable', 'defenseless', 'unsafe', 'threatened', 'insecure', 'dismayed', 'hysterical', 'aghast', 'suspicious', 'uncomfortable', 'unnerved', 'unsure', 'wary', 'alarmed', 'disturbed', 'panic', 'timid', 'troubled', 'upset', 'haunted', 'jittery', 'menaced', 'perturbed', 'rattled', 'shocked', 'spooked', 'stricken', 'timorous', 'apprehensive', 'disquieted', 'distraught', 'frantic', 'faint-hearted'],
    'surprise': ['startled', 'jolted', 'shocked', 'amazed', 'astonished', 'dumbfounded', 'flabbergasted', 'stunned', 'stupefied', 'taken-aback', 'thunderstruck', 'surprised', 'disbelieving', 'incredulous', 'bewildered', 'confused', 'disoriented', 'perplexed', 'puzzled', 'awed', 'speechless', 'mesmerized', 'enchanted', 'captivated', 'spellbound', 'fascinated', 'intrigued', 'absorbed', 'gripped', 'engaged', 'riveted', 'hooked', 'fixed', 'dazzled', 'bedazzled', 'blinded', 'gobsmacked', 'giddy', 'light-headed', 'woozy', 'dizzy', 'discombobulated', 'reeling', 'uncertain', 'unsettled', 'thrown-off', 'caught-off-guard', 'unclear', 'confounded', 'staggered', 'shaken-up', 'knocked-for-a-loop', 'flustered', 'flummoxed', 'nonplussed', 'shaken']
}

#m5 is the speech emotion AI, affectnet is the video facial emotion AI
m5_to_affect_net_converter_dict = {
    "sadness": "sadness",
    "joy": "Happiness",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise"
}

emotions_one_word_actions = {
'sadness': 'sadness-7ib4tuu',
    'joy': 'joy-incoyf9',
    'anger': 'anger-tcnuo8v',
    'fear': 'fear-vfym9t7',
    'surprise': 'surprise-oellmme'
}

#affectnet emotions to accept or ignore
#affectnet_emotion_ignore = ['Neutral','Disgust']
#affectnet_emotion_accept = ['Happiness','sadness','surprise','fear','anger']

#aws rekognition emotions to accept or ignore
#rekognition_emotion_ignore = ["CONFUSED", "DISGUSTED", "CALM", "UNKNOWN"] 
#rekognition_emotion_accept = ['HAPPY' , 'SAD' , 'ANGRY', 'CONFUSED' , 'SURPRISED', 'FEAR']


#'output_youtube.json'
def transcript_to_df_youtube(transcript:str):

    with open(transcript, 'r') as f:
        content = f.read()

    #prep data to df
    df = df = pd.read_json(content, orient= 'columns')

    df['time_end'] = df['time_start'] + df['duration']

    # deal with echo (a line end later than the start line)
    for index,row in df.iterrows():
        try:
            if df.loc[index,'time_end'] > df.loc[index+1,'time_start']:
                df.loc[index,'time_end'] = df.loc[index+1,'time_start']
        except KeyError:
            pass


    #deal with transcript line that has 0 duration (transcript error?)
    for index,row in df.iterrows():
        row_duration = row['duration']
        if row['duration'] < 0 :
            try:
                df.loc[index,'time_end'] = df.loc[index+1,'time_end']
            except KeyError:
                df.loc[index,'time_end'] = 0.5

    # determine which part of the video is quiet so that stock video length will be accurate.
    df['slient_between_lines'] = df['time_start'] - df['time_end'].shift(1)
    df['slient_between_lines'].fillna(0, inplace=True)

    #with open('summary.json', 'w') as out:
    #    json.dump(json.loads(df.to_json()), out)  

    return df


#'transcript.json'
def transcript_to_df(transcript:json):

    with open(transcript, 'r') as f:
        content = f.read()

    #prep data to df
    my_dict = json.loads(content)
    df = pd.json_normalize(my_dict["results"])
    df.drop('final', axis=1, inplace=True)
    alt_df = pd.json_normalize(df['alternatives'].apply(pd.Series).stack().reset_index(drop=True))
    alt_df['time_start'] = alt_df['timestamps'].apply(lambda x: x[0][1])
    alt_df['time_end'] = alt_df['timestamps'].apply(lambda x: x[-1][2])
    alt_df.drop('timestamps', axis=1, inplace=True)

    # deal with echo (a line end later than the start line)
    shifted_time_end = alt_df['time_end'].shift(1)
    alt_df['time_start'] = alt_df['time_start'].clip(lower=shifted_time_end)

    # determine which part of the video is quiet so that stock video length will be accurate.
    alt_df['slient_between_lines'] = alt_df['time_start'] - alt_df['time_end'].shift(1)
    alt_df['slient_between_lines'].fillna(0, inplace=True)

    #with open('transcript_df.json', 'w') as out:
    #    out.write(alt_df.to_json(orient = 'columns'))

    return alt_df.to_json(orient = 'columns')

def check_stopword(transcript:str):
    stop_words = set(stopwords.words("english"))
    
    #remove puncuation and spaces and change word to lower before checking with stop word dicts
    if ''.join(re.findall(r'\w+', transcript)).strip().lower() in stop_words:
        return True
    return False

def return_non_stopword(transcript:str):
    
    stop_words = set(stopwords.words("english"))
    
    # Split transcript into individual words and remove punctuation and spaces
    words = re.findall(r'\w+', transcript)
    #print(f'words ares {words}')

    if len(words) == 0:
        return ' '.join('')

    # Filter out stop words
    non_stopwords = [word for word in words if word.strip().lower() not in stop_words]
    #print(non_stopwords)
    # Join remaining words into a single string and return it
    return ' '.join(non_stopwords)

#transcript.json
def timestamp_to_df(timestamp:json):

    with open(timestamp, 'r') as f:
        content = f.read()

    #prep data to df
    my_dict = json.loads(content)
    print(my_dict)
    df = pd.DataFrame(my_dict)


    df = df.rename(columns={'word':'transcript','start': 'time_start','end':'time_end'})
    
    # deal with echo (a line end later than the start line)
    shifted_time_end = df['time_end'].shift(1)
    df['time_start'] = df['time_start'].clip(lower=shifted_time_end)

    df['stopwords'] = df['transcript'].apply(lambda x: check_stopword(x))

    # determine which part of the video is quiet so that stock video length will be accurate.
    df['slient_between_lines'] = df['time_start'] - df['time_end'].shift(1)
    df['slient_between_lines'].fillna(0, inplace=True)

    #with open('transcript_df.json', 'w') as out:
    #    out.write(df.to_json())
    
    return df

#transcript_df_adjusted_list.json
def smart_reframe_df(df:object, words_per_row):

    collapsed_df = pd.DataFrame(columns=['transcript', 'time_start', 'time_end','probability','stopwords','slient_between_lines'])

    try:
        while True:
            fullstop_index = df[df['transcript'].str.contains('[\.,?!]')].index[0] + 1
            print(fullstop_index)
            num_to_seperate = words_per_row
            if fullstop_index > num_to_seperate:
                remainder = fullstop_index%num_to_seperate
                loop_num = fullstop_index//num_to_seperate
                print(f'remainder,loop_num {remainder},{loop_num}')

                for loop in range(loop_num):
                    print(f'loop num {loop}')

                    #check for last loop
                    if (loop == loop_num-1) and (remainder > 0):
                        print(f'at remainder loop {loop}...')
                        collapsed_row = pd.DataFrame({
                            'transcript': [df['transcript'][:remainder].tolist()],
                            'time_start': [df['time_start'][:remainder].tolist()],
                            'time_end': [df['time_end'][:remainder].tolist()],
                            'probability': [df['probability'][:remainder].tolist()],
                            'stopwords': [df['stopwords'][:remainder].tolist()],
                            'slient_between_lines': [df['slient_between_lines'][:remainder].tolist()]
                        })
                        #print(f'remainder loop {collapsed_row}')
                        df = df[remainder:]
                        df.reset_index(drop=True, inplace=True)
                        
                    else:
                        print(f'at loop {loop}...')
                        collapsed_row = pd.DataFrame({
                            'transcript': [df['transcript'][:num_to_seperate].tolist()],
                            'time_start': [df['time_start'][:num_to_seperate].tolist()],
                            'time_end': [df['time_end'][:num_to_seperate].tolist()],
                            'probability': [df['probability'][:num_to_seperate].tolist()],
                            'stopwords': [df['stopwords'][:num_to_seperate].tolist()],
                            'slient_between_lines': [df['slient_between_lines'][:num_to_seperate].tolist()]
                        })
                        #print(f'seperate loop {collapsed_row}')
                        df = df[num_to_seperate:]
                        df.reset_index(drop=True, inplace=True)

                    collapsed_df = pd.concat([collapsed_df,collapsed_row])
                    #print(collapsed_df)

            else:
                print(f'fullstop_index len is now {fullstop_index}')
                collapsed_row = pd.DataFrame({
                    'transcript': [df['transcript'][:fullstop_index].tolist()],
                    'time_start': [df['time_start'][:fullstop_index].tolist()],
                    'time_end': [df['time_end'][:fullstop_index].tolist()],
                    'probability': [df['probability'][:fullstop_index].tolist()],
                    'stopwords': [df['stopwords'][:fullstop_index].tolist()],
                    'slient_between_lines': [df['slient_between_lines'][:fullstop_index].tolist()]
                })
                #print(f'fullstop loop {collapsed_row}')
                df = df[fullstop_index:]
                #print(f'df is {df} and index is {fullstop_index}')
                df.reset_index(drop=True, inplace=True)
                collapsed_df = pd.concat([collapsed_df,collapsed_row])
                

    except (ValueError,IndexError):
        if IndexError:
            try:
                print('indexerror')
                #print(df)
                fullstop_index = df[:].index[-1] + 1
                collapsed_row = pd.DataFrame({
                    'transcript': [df['transcript'][:fullstop_index].tolist()],
                    'time_start': [df['time_start'][:fullstop_index].tolist()],
                    'time_end': [df['time_end'][:fullstop_index].tolist()],
                    'probability': [df['probability'][:fullstop_index].tolist()],
                    'stopwords': [df['stopwords'][:fullstop_index].tolist()],
                    'slient_between_lines': [df['slient_between_lines'][:fullstop_index].tolist()]
                })

                df = df[fullstop_index:]
                df.reset_index(drop=True, inplace=True)
                collapsed_df = pd.concat([collapsed_df,collapsed_row])
            except IndexError:
                pass
        print(f'job done')

    collapsed_df.reset_index(drop=True, inplace=True)
    print(collapsed_df)



    #adjusted_collapsed_df = pd.DataFrame(columns=['transcript', 'time_start', 'time_end','probability','stopwords','slient_between_lines'])

    for idx,rows in collapsed_df.iterrows():
        """
        Transform collapsed df column data to a specific format
        
        """
        print(f'now at idx {idx}')

        # Join transcript into a single string
        collapsed_df.loc[idx,'transcript'] = ''.join(collapsed_df.loc[idx,'transcript'])

        # Extract first and last values from time_start and time_end
        collapsed_df.loc[idx,'time_start'] = collapsed_df.loc[idx,'time_start'][0]
        collapsed_df.loc[idx,'time_end'] = collapsed_df.loc[idx,'time_end'][-1]

        # Calculate the average of probability
        collapsed_df.loc[idx,'probability'] = sum(collapsed_df.loc[idx,'probability']) / len(collapsed_df.loc[idx,'probability'])


        # Extract first value from silent_between_lines
        collapsed_df.loc[idx,'slient_between_lines'] = collapsed_df.loc[idx,'slient_between_lines'][0]
        #new_row = pd.DataFrame({'transcript': transcript, 'time_start': time_start, 'time_end': time_end, 'probability': probability_avg, 'stopwords': stopwords, 'slient_between_lines': slient_between_lines})
        #adjusted_collapsed_df = pd.concat([adjusted_collapsed_df,new_row])
        #print("adjusted_collapsed_df",adjusted_collapsed_df)

    #with open('transcript_df_adjusted_list.json', 'w') as out:
    #    out.write(adjusted_collapsed_df.to_json())

    return collapsed_df
    #for idx in range(len(df)):


def collapse_trailer_df(df:object):
    df = df.dropna(subset=['filename_celeb_vertical'])
    df = df.reset_index(drop=True)

def trailer_df_combine(trailerinfo_location = r'trailer', filename_to_save='merged.json'):

    df_list = []
    
    for file_name in os.listdir(trailerinfo_location):
        print(file_name)
        if file_name.endswith('.json'):
            file_path = os.path.join(trailerinfo_location, file_name)
            df = pd.read_json(file_path,orient= 'columns')
            df['filename'] = file_name
            df_list.append(df)

    merged_df = pd.concat(df_list, ignore_index=True)

    #filter emotion 'Neutral','Disgust' because they are not in m5 speech langaugae AI
    filter_merged_df = merged_df.loc[~merged_df['emotion'].isin(['Neutral', 'Disgust']),:]
    filter_merged_df['emotion'] = filter_merged_df['emotion'].replace('Happiness', 'joy')
    print(f'number of rows got filtered are {len(merged_df)/len(filter_merged_df)}')

    filter_merged_df.to_json(f"{trailerinfo_location}/{filename_to_save}")

