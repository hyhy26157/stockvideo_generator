This is a speech to video generator. Video are query based on the sentiment of the transcript of the speech.mp4.
To start, 
1) put in the speech.mp3 file at the root directory 
2) pip install -r requirements.txt
3) Create your own folder called Keys. Create a keys.py file. You will need openAI, pexel and huggingface API.
*Note: See keys.example for more information.
4) start main.py
vola! finaalised video will be in the final_clip folder.

*Note: Delete all the json file and .mp4 files (root, final_clip folder,adjusted_framerate_video_folder) 
before creating another video to avoid errors.


If you want to adjust the video parameters, you can use the config.yaml file

*Note: By default, video generated are in Shorts format bias towards female appearance. You can change the parameters, depend on your requirement. Read up on pexels API for more information.

