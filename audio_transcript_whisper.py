import json
import whisper
import yaml

#transcribe audio using whisper model
def audio_transcript_whisper():

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model = whisper.load_model("base")
    result = model.transcribe(config['speech_file'], word_timestamps=True)

    print(result["segments"][0]['words'])

    segments = result["segments"]
    segment_list = []
    for segment in segments:
        segment_list.append(segment["words"])

    flat_list = [item for sublist in segment_list for item in sublist]

    with open(config['speech_transcript_json_raw'], 'w') as out:
        json.dump(flat_list, out)

  
