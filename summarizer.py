from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def preprocess_text(file_path, speakers=("Senator", "Mark")):
    """
    Separates the dialogue into two parts for the specified speakers.

    Args:
        file_path (str): Path to the transcript text file.
        speakers (tuple): Tuple containing the names of the two speakers.

    Returns:
        dict: A dictionary with speaker names as keys and their respective dialogue as values.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        transcript = file.read()

    separated_text = {speaker: [] for speaker in speakers}
    lines = transcript.splitlines()

    for line in lines:
        line = line.strip()
        for speaker in speakers:
            if line.startswith(f"{speaker}:"):
                dialogue = line[len(f"{speaker}:"):].strip()
                separated_text[speaker].append(dialogue)
                break

    separated_text = {speaker: " ".join(dialogues) for speaker, dialogues in separated_text.items()}

    return separated_text

file_path = "transcript.txt"

speaker_texts = preprocess_text(file_path, speakers=("Senator", "Mark"))

senator_text = speaker_texts["Senator"]
mark_text = speaker_texts["Mark"]

model_name = "MingZhong/DialogLED-large-5120"  
tokenizer_dl = AutoTokenizer.from_pretrained(model_name)
model_dl = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_summary(text, max_input_length=512, max_summary_length=150):
    inputs = tokenizer_dl(text, return_tensors="pt", truncation=True, max_length=max_input_length)
    summary_ids = model_dl.generate(inputs["input_ids"], max_length=max_summary_length, min_length=50, length_penalty=2.0, num_beams=5, early_stopping=True)
    return tokenizer_dl.decode(summary_ids[0], skip_special_tokens=True)

def split_text(text, max_tokens=512):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def summary_chunks(text, max_length=512):
  chunks = split_text(text)
  summaries = [generate_summary(chunk) for chunk in chunks]
  final_summary = "\n".join(summaries)
  return final_summary

senator_summary = summary_chunks(senator_text)
mark_summary = summary_chunks(mark_text)

print('Senator Summary:', senator_summary)
print('Mark Summary:', mark_summary)