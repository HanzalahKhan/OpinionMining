import assemblyai as aai

aai.settings.api_key = "ddbae00caade4d5086c70329b6470f1a"
transcriber = aai.Transcriber()

config = aai.TranscriptionConfig(
    speaker_labels=True
)

transcript = aai.Transcriber().transcribe("vd.mp4", config)
   
speaker_names = {
    "A": "Senator",
    "B": "Mark",
    # Add more mappings as needed
}

# for utterance in transcript.utterances:
#     speaker_tag = utterance.speaker
#     speaker_name = speaker_names.get(speaker_tag, "Unknown Speaker")

#     print(f"{speaker_name}: {utterance.text}")

# Assuming 'transcript' and 'speaker_names' are already defined
output_file = "transcript.txt"  # Specify the file name

print('reading lines')
# Use a list to collect all lines
lines = []
for utterance in transcript.utterances:
    speaker_tag = utterance.speaker
    speaker_name = speaker_names.get(speaker_tag, "Unknown Speaker")
    lines.append(f"{speaker_name}: {utterance.text}")

print(f"Saving transcript to {output_file}...")
# Write all lines at once to the file
with open(output_file, "w", encoding="utf-8") as file:
    file.write("\n".join(lines))

print(f"Transcript saved successfully to {output_file}")
