# youtube upload api
from youtube_upload.client import YoutubeUploader


uploader = YoutubeUploader(secrets_file_path='client_secrets.json')

uploader.authenticate()


# Video options
options = {
    "title" : "Example title", # The video title
    "description" : "Example description", # The video description
    "tags" : ["tag1", "tag2", "tag3"],
    "categoryId" : "22",
    "privacyStatus" : "private", # Video privacy. Can either be "public", "private", or "unlisted"
    "kids" : False # Specifies if the Video if for kids or not. Defaults to False.
    #"thumbnailLink" : "https://cdn.havecamerawilltravel.com/photographer/files/2020/01/youtube-logo-new-1068x510.jpg" # Optional. Specifies video thumbnail.
}

# upload video
uploader.upload('index 0_v2.mp4', options) 

#close the authetication
uploader.close()