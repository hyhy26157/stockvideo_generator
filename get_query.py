import openai
from Keys.keys import openai_api_key, huggingfacekey
import time
import json
import pandas as pd
import re
import requests
from transcript_to_dataframe import return_non_stopword
import random
import time
import torch
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModelWithLMHead
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

#engine is t5 or chatgpt
def summerisation_general(df:object,num_char_prompt:int = config['num_char_prompt'],engine=config['summarisation_engine']):

    # Open the df for reading
    df['query'] = ''
    querywords = []

    print(f'the columns of the df are {df.columns}')
    # summary transcript line into 1 word query
    word_cummu_length = 0
    word_cummu  = ""
    index_skipped = 0

    # to exit loop even when char limit is not satisfied
    last_index = df.index[-1]


    #set stop words
    stop_words = set(stopwords.words('english'))

    # https://huggingface.co/mrm8488/t5-base-finetuned-emotion
    #transform text to 6 emotions
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    df = df.reset_index(drop=True)
    
    for index,row in df.iterrows():
        print(f'starting index {index}')

        #to cummulate transcripts for prompting
        if word_cummu_length < num_char_prompt:

            transcript = return_non_stopword(row['transcript'])
            
            #to cummulate transcripts for prompting
            word_cummu_length += len(transcript)
            word_cummu += transcript + " "
            
            if index != last_index:
                index_skipped += 1
                continue
            else:
                pass
        else:
            prev_word = df.loc[index-1,'transcript']
            current_word = df.loc[index,'transcript']
            print(f'prev_word {prev_word}')
            print(f'index is at {index} and the transcript is {transcript}')
            #print(f'the last index transcript is {prev_word}')
            if ('.' in prev_word) or ('?' in prev_word) or ('!' in prev_word):
                pass
            else:
                #to cummulate transcripts to a meaningful stop for better prompting result
                transcript = return_non_stopword(row['transcript'])

                word_cummu_length += len(transcript)
                word_cummu += transcript + " "
                print(f'word_cummu {word_cummu}')
                if index != last_index:
                    index_skipped += 1
                    continue
                else:
                    pass

        if (word_cummu_length >= num_char_prompt) or (index == last_index):
        
            if engine =='openai':
                # Remove periods
                periods = re.sub(r'(\.)+', '', openai_query(word_cummu,'transcript'))
                # Extract last word
                try:
                    last_word = re.findall(r'\b\w+\b\s*$', periods)[0]
                except IndexError:
                    last_word = re.findall(r'\b\w+\b\s*$', periods)
                # Remove newlines
                try:
                    newlines = re.sub(r'(\n)+', '', last_word)
                except TypeError:
                    newlines = re.sub(r'(\n)+', '', last_word)[0]

                print(f'\nThe one word summmary post-processed is \n{str(newlines)}')

                for i in range(index_skipped,-1,-1):
                    # Assign to 'query' column
                    df.loc[(index - i), 'query'] = str(newlines)
                
                #reset
                index_skipped = 0
                word_cummu_length = 0
                word_cummu  = ""

                #prevent api overload
                time.sleep(2)
            
            elif engine == 't5':
                emotion = get_emotion(word_cummu,tokenizer,model)

                emotion = emotion.split()[-1]

                print(f'\nThe one word summmary post-processed is \n{str(emotion)}')

                for i in range(index_skipped,-1,-1):
                    # Assign to 'query' column
                    df.loc[(index - i), 'query'] = str(emotion)

                #reset
                index_skipped = 0
                word_cummu_length = 0
                word_cummu  = ""
            
        with open('summary.json', 'w') as out:
            out.writelines(df.to_json(orient = 'columns'))

    #df['query'] = df['query'].replace('',None)
    df = df.fillna(method='backfill')

    with open('summary.json', 'w') as out:
            out.write(df.to_json(orient = 'columns'))

#give synomyns
def thesaurus_webster(word:str) -> list:

    IT_key = '865bd34d-9273-4d56-9e6a-1651214e2aa4'
    api_url  = f'https://www.dictionaryapi.com/api/v3/references/thesaurus/json/{word}?key={IT_key}'
    response = requests.get(api_url)
    #print(response,'\n\n\n\n\n')
    if response.status_code == requests.codes.ok:
        #print(response.text)
        info = response.text
        info = json.loads(info)
        try:
            stems_list = info[0]['meta']['stems']
            syns_list = info[0]['meta']['syns']
            syns_list = [item for sublist in syns_list for item in sublist]
            ants_list = info[0]['meta']['ants']
            ants_list = [item for sublist in ants_list for item in sublist]
            alt_words_list = stems_list + syns_list + ants_list
            alt_words_list = sorted(alt_words_list, key=len)
            #search by stems,syns,ants order

            print(f'alt words are {alt_words_list}')
        except (TypeError, IndexError):
            alt_words_list = info
            print(f'alt words are {alt_words_list}')
        return alt_words_list
    else:
        return [word]

def openai_query(word_cummu:str,summary_type:str):
    openai.api_key  = openai_api_key

    if summary_type =='transcript':
        words = f"\ndescribe this sentence with one emotion word: \n ({word_cummu}) \n "
    elif summary_type =='movie':
        words = f"\nsummarise what this movie text is about in only one word: \n ({word_cummu}) \n "

    print(f'{words}')
    completion = openai.Completion.create(engine="text-davinci-003", prompt=words)
    one_word_summary = completion['choices'][0]['text']

    return str(one_word_summary)

#sadness,joy,anger,fear,surprise. ignored love because rekognition doesn't do love
def get_emotion(text,tokenizer,model):
  input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

  output = model.generate(input_ids=input_ids,
               max_length=2)
  
  dec = [tokenizer.decode(ids) for ids in output]
  label = dec[0]

  if label == 'love':
      label = 'Joy'
  return label

def emotion_to_action(df,video_interval:int = config['emotion_change_frequency']):

    emotions_one_word_actions_dict = {
    'sadness': ['crying', 'mourning', 'weeping', 'grieving', 'depressed', 'tearing', 'heartbroken', 'hopeless', 'lonely', 'abandoned', 'neglected', 'rejected','unhappy', 'miserable' ,'disappointed', 'disheartened','Break up', 'Failing', 'in pain', 'suffering', 'disappointed'],
        'joy': ['smiling', 'laughing', 'dancing', 'celebrating','excited', 'happy', 'grateful', 'thankful', 'satisfied', 'peaceful', 'motivated', 'confident', 'optimistic', 'energized',  'dance', 'sing', 'laugh', 'smile', 'enjoying'],
        'anger': ['angry','yelling', 'shouting', 'screaming','mad', 'annoyed', 'irritated', 'frustrated', 'hit things', 'kick things', 'punching','slamming', 'scolding'],
        'fear': ['freezing', 'drowning','scared', 'shivering', 'hiding', 'avoiding', 'fearing', 'panicking', 'terrifying', 'frightening', 'scaring', 'horrifying','anxious', 'nervous', 'worring', 'scared', 'trapped', 'threatened', 'insecure','suspecting', 'uncomfortable', 'unsure', 'panic', 'shocked'],
        'surprise': ['shocked', 'surprised']
    }

    # group query when timing is >5 seconds
    subgroup_index_list = subgrouping_df(df, duration_threshold  = config['emotion_change_frequency'])
    subgroup_first_index = [sublist[0] for sublist in subgroup_index_list]
    print("len(subgroup_index_list)",len(subgroup_index_list))
    print("subgroup_first_index",subgroup_first_index)
    skipped_index = 0
    for index,row in df.iterrows():
        print("skipped_index",skipped_index)
        if skipped_index != 0:
            skipped_index -= 1
            continue
        

        current_query = row['query']
        #print(f'there are no gender and current query is {current_query}')

        # select a value from the list associated with the selected key randomly
        random_action = random.choice(emotions_one_word_actions_dict[current_query])
        emotions_one_word_actions_dict[current_query].remove(random_action)
        #print('random_action',random_action)
        #print(f"is index {index} in subgroup {subgroup_first_index}")
        if index in subgroup_first_index:
                position = subgroup_first_index.index(index)
                #print("index,position",index,position)
                #print("subgroup_first_index[position]",subgroup_index_list[position])
                skipped_index -= 1
                for inner_index in subgroup_index_list[position]:
                    #print("inner_index",inner_index)
                    df.loc[inner_index,'query'] = random_action
                    #print("df.loc[inner_index,'query']",df.loc[inner_index,'query'])
                    skipped_index +=1
        #print("df.loc[min(subgroup_inde", df.loc[min(subgroup_index_list[subgroup_index_list_counter]):max(subgroup_index_list[subgroup_index_list_counter]),'query'])

    return df, emotions_one_word_actions_dict

#take in df query column and duration column with a 5 seconds therehold
def subgrouping_df(df, duration_threshold  = config['emotion_change_frequency']):

    def subgroup_duration(df_duration, groups):
        subgroups = []
        total_duration = 0

        for group in groups:
            #print('group',group)
            subgroup_duration = df_duration[group[0]:group[1]+1]
            #print('group[1]',group[1])
            #print('len subgroup_duration',len(subgroup_duration))
            subgroup_index = list(range(group[0], group[1]+1))
            #print("len subgroup_index",len(subgroup_index))
            #print('lens',len(subgroup_duration),len(subgroup_index))

            initial_idx = min(subgroup_index)
            end_idx = min(subgroup_index)
            #print('initial_idx end_idx',initial_idx,end_idx)
            #print('starting catch_last_group',catch_last_group)
            for idx,duration in zip(subgroup_index, subgroup_duration):
                total_duration += duration
                #print(initial_idx,end_idx,total_duration, num_of_loop)
                if total_duration >= duration_threshold:
                    subgroup_indices = list(range(initial_idx, end_idx+1))
                    #print('subgroup_indices',subgroup_indices)
                    subgroups.append(subgroup_indices)
                    #print(subgroups)
                    total_duration = 0
                    initial_idx = end_idx + 1
                    end_idx += 1
                else:
                    end_idx += 1
            #print('subgroups[-1][-1] != group[1]',subgroups[-1][-1],group[1])
            if subgroups[-1][-1] != group[1]:
                #print('catch_last_group',catch_last_group,group[1])
                subgroup_indices = list(range(subgroups[-1][-1]+1, group[1] +1))
                #print(subgroup_indices)
                amended_last_subgroups = subgroups[-1] + subgroup_indices
                subgroups.pop(-1)
                subgroups.append(amended_last_subgroups)
                total_duration = 0
                #print('subgroups amended', subgroups)

        print(subgroups)
        return subgroups

    groups = df.groupby(df['query']).apply(lambda x: [[g[0], g[-1]] for g in pd.DataFrame({'idx': x.index, 'diff': x.index.to_series().diff().fillna(1).ne(1).cumsum()}).groupby('diff')['idx'].apply(list)])
    #print(groups)
    result = sorted([t for sublist in groups.values for t in sublist])
    subgroups_filtered = subgroup_duration(df['duration'],result)
    #print(subgroups_filtered,'\n')
    return subgroups_filtered
