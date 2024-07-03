
# required modules
import os
import numpy as np
from pytube import YouTube
from pydub import AudioSegment
from faster_whisper import WhisperModel
import tempfile
import tkinter as tk
from googletrans import Translator
import nltk
from docx import Document
from PyPDF2 import PdfReader
import threading
import time
from datetime import datetime
from gensim.summarization.summarizer import summarize


'''

800 MB for the whisper "medium" model.
this project requires 1-2 GB RAM 

'''

print("start time : ", datetime.now().time())


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


model_size = "medium"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

nltk.download('punkt')


def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def download_youtube_audio(youtube_url):
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    audio_stream.download(output_path=os.path.dirname(temp_file.name), filename=os.path.basename(temp_file.name))
    return temp_file.name


def convert_audio_to_wav(audio_file, target_sample_rate=16000):
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(target_sample_rate).set_channels(1)
    temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.export(temp_wav_file.name, format="wav")
    return temp_wav_file.name


def transcribe_audio(wav_file):
    transcriptions = []
    audio = AudioSegment.from_wav(wav_file)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples = samples / np.iinfo(np.int16).max  # Normalize samples to float32 range

    segments, info = model.transcribe(samples, beam_size=5)

    for segment in segments:
        transcription = segment.text
        transcriptions.append(transcription)

    return transcriptions


'''def summarize(text, num_sentences):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return ' '.join(str(sentence) for sentence in summary)'''
def summarize_text(text, wc):
    # Gensim's summarize function takes a ratio or word count for summarization
    return summarize(text, word_count = wc )  # Assuming average sentence length of 20 words




def read_docx_file(file_path):
    doc = Document(file_path)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)


def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    full_text = []
    for page in reader.pages:
        full_text.append(page.extract_text())
    return '\n'.join(full_text)


def read_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.txt':
        return read_txt_file(file_path)
    elif file_extension == '.docx':
        return read_docx_file(file_path)
    elif file_extension == '.pdf':
        return read_pdf_file(file_path)
    else:
        raise ValueError("Unsupported file type: {}".format(file_extension))


def translate_text_google(text, dest_language):
    translator = Translator()
    try:
        translation = translator.translate(text, dest=dest_language)
        if translation and translation.text:
            return translation.text
        else:
            raise ValueError("Empty translation response")
    except Exception as e:
        print(f"Translation error: {e}")
        return None


class TypingThread(threading.Thread):
    def __init__(self, widget, text, delay=50):
        super().__init__()
        self.widget = widget
        self.text = text
        self.delay = delay
        self._stop_event = threading.Event()

    def run(self):
        self.widget.delete(1.0, tk.END)
        for char in self.text:
            if self._stop_event.is_set():
                break
            try:
                self.widget.insert(tk.END, char)
                self.widget.see(tk.END)
                time.sleep(self.delay / 1000.0)
            except tk.TclError:
                break  

    def stop(self):
        self._stop_event.set()


def show_transcription_page():
    for widget in main_frame.winfo_children():
        widget.destroy()

    def on_transcribe_click():
        global typing_thread
        audio_file = youtube_url_entry.get()
        segments, info = model.transcribe(audio_file, beam_size=5)

        transcriptions = []
        for segment in segments:
            transcription = segment.text
            transcriptions.append(transcription)

        typing_thread = TypingThread(transcription_text, '\n'.join(transcriptions))
        typing_thread.start()

    def on_stop_click():
        if typing_thread:
            typing_thread.stop()

    def on_clear_click():
        transcription_text.delete(1.0, tk.END)

    youtube_url_label = tk.Label(main_frame, text="Enter file path (only mp3 file):", bg=main_bg_color, fg=main_text_color)
    youtube_url_label.pack(pady=5)
    youtube_url_entry = tk.Entry(main_frame, width=50)
    youtube_url_entry.pack(pady=5)

    button_frame = tk.Frame(main_frame, bg=main_bg_color)
    button_frame.pack(pady=10)

    transcribe_button = tk.Button(button_frame, text="Transcribe", command=on_transcribe_click, bg=button_color,
                                  fg=button_text_color)
    transcribe_button.pack(side=tk.LEFT, padx=5)
    stop_button = tk.Button(button_frame, text="Stop", command=on_stop_click, bg=button_color, fg=button_text_color)
    stop_button.pack(side=tk.LEFT, padx=5)
    clear_button = tk.Button(main_frame, text="Clear", command=on_clear_click, bg=button_color, fg=button_text_color)
    clear_button.pack(pady=5)

    transcription_text = tk.Text(main_frame, wrap=tk.WORD, height=20, width=80, bg=text_bg_color, fg=text_fg_color)
    transcription_text.pack(pady=5)


def show_summary_page():
    for widget in main_frame.winfo_children():
        widget.destroy()

    def on_summarize_click():
        global typing_thread
        file_path = file_path_entry.get()
        num_sentences = int(num_sentences_entry.get())
        text = read_file(file_path)
        summary_text = summarize_text(text, num_sentences)
        typing_thread = TypingThread(summary_text_widget, summary_text)
        typing_thread.start()

    def on_stop_click():
        if typing_thread:
            typing_thread.stop()

    def on_clear_click():
        summary_text_widget.delete(1.0, tk.END)

    file_path_label = tk.Label(main_frame, text="File Path (text/pdf/docx file):", bg=main_bg_color, fg=main_text_color)
    file_path_label.pack(pady=5)
    file_path_entry = tk.Entry(main_frame, width=50)
    file_path_entry.pack(pady=5)
    num_sentences_label = tk.Label(main_frame, text="In how many words you want to summarize :", bg=main_bg_color, fg=main_text_color)
    num_sentences_label.pack(pady=5)
    num_sentences_entry = tk.Entry(main_frame, width=10)
    num_sentences_entry.pack(pady=5)

    button_frame = tk.Frame(main_frame, bg=main_bg_color)
    button_frame.pack(pady=10)

    summarize_button = tk.Button(button_frame, text="Summarize", command=on_summarize_click, bg=button_color,
                                 fg=button_text_color)
    summarize_button.pack(side=tk.LEFT, padx=5)
    stop_button = tk.Button(button_frame, text="Stop", command=on_stop_click, bg=button_color, fg=button_text_color)
    stop_button.pack(side=tk.LEFT, padx=5)
    clear_button = tk.Button(main_frame, text="Clear", command=on_clear_click, bg=button_color, fg=button_text_color)
    clear_button.pack(pady=5)

    summary_text_widget = tk.Text(main_frame, wrap=tk.WORD, height=20, width=80, bg=text_bg_color, fg=text_fg_color)
    summary_text_widget.pack(pady=5)


def show_translation_page():
    for widget in main_frame.winfo_children():
        widget.destroy()

    def on_translate_click():
        global typing_thread
        text = read_txt_file(file_path_entry.get())
        translated_summary = translate_text_google(text, language_entry.get())
        if translated_summary:
            typing_thread = TypingThread(translation_text_widget, translated_summary)
            typing_thread.start()
            with open("translated_summary.txt", 'w', encoding='utf-8') as file:
                file.write(translated_summary)

    def on_stop_click():
        if typing_thread:
            typing_thread.stop()

    def on_clear_click():
        translation_text_widget.delete(1.0, tk.END)

    file_path_label = tk.Label(main_frame, text="File Path (text file only):", bg=main_bg_color, fg=main_text_color)
    file_path_label.pack(pady=5)
    file_path_entry = tk.Entry(main_frame, width=50)
    file_path_entry.pack(pady=5)
    language_label = tk.Label(main_frame, text="Destination Language (e.g., 'hi' for Hindi):", bg=main_bg_color,
                              fg=main_text_color)
    language_label.pack(pady=5)
    language_entry = tk.Entry(main_frame, width=10)
    language_entry.pack(pady=5)

    button_frame = tk.Frame(main_frame, bg=main_bg_color)
    button_frame.pack(pady=10)

    translate_button = tk.Button(button_frame, text="Translate", command=on_translate_click, bg=button_color,
                                 fg=button_text_color)
    translate_button.pack(side=tk.LEFT, padx=5)
    stop_button = tk.Button(button_frame, text="Stop", command=on_stop_click, bg=button_color, fg=button_text_color)
    stop_button.pack(side=tk.LEFT, padx=5)
    clear_button = tk.Button(main_frame, text="Clear", command=on_clear_click, bg=button_color, fg=button_text_color)
    clear_button.pack(pady=5)

    translation_text_widget = tk.Text(main_frame, wrap=tk.WORD, height=20, width=80, bg=text_bg_color, fg=text_fg_color)
    translation_text_widget.pack(pady=5)


def show_youtube_transcription_page():
    for widget in main_frame.winfo_children():
        widget.destroy()

    def on_transcribe_click():
        global typing_thread
        youtube_url = youtube_url_entry.get()
        audio_file = download_youtube_audio(youtube_url)
        wav_file = convert_audio_to_wav(audio_file)
        transcriptions = transcribe_audio(wav_file)
        with open("yt_text.txt", "w") as f:
            for transcription in transcriptions:
                f.write(transcription + "\n")
        typing_thread = TypingThread(transcription_text, '\n'.join(transcriptions))
        typing_thread.start()
        os.remove(audio_file)
        os.remove(wav_file)

    def on_stop_click():
        if typing_thread:
            typing_thread.stop()

    def on_clear_click():
        transcription_text.delete(1.0, tk.END)

    youtube_url_label = tk.Label(main_frame, text="YouTube URL (audio should be english)", bg=main_bg_color, fg=main_text_color)
    youtube_url_label.pack(pady=5)
    youtube_url_entry = tk.Entry(main_frame, width=50)
    youtube_url_entry.pack(pady=5)

    button_frame = tk.Frame(main_frame, bg=main_bg_color)
    button_frame.pack(pady=10)

    transcribe_button = tk.Button(button_frame, text="Transcribe", command=on_transcribe_click, bg=button_color,
                                  fg=button_text_color)
    transcribe_button.pack(side=tk.LEFT, padx=5)
    stop_button = tk.Button(button_frame, text="Stop", command=on_stop_click, bg=button_color, fg=button_text_color)
    stop_button.pack(side=tk.LEFT, padx=5)
    clear_button = tk.Button(main_frame, text="Clear", command=on_clear_click, bg=button_color, fg=button_text_color)
    clear_button.pack(pady=5)

    transcription_text = tk.Text(main_frame, wrap=tk.WORD, height=20, width=80, bg=text_bg_color, fg=text_fg_color)
    transcription_text.pack(pady=5)


# Colors
main_bg_color = "#2B2B2B"
main_text_color = "#F5F5F5"
button_color = "#3A3A3A"
button_text_color = "#FFFFFF"
text_bg_color = "#1E1E1E"
text_fg_color = "#D4D4D4"
print("display time : ", datetime.now().time())

# main window
root = tk.Tk()
root.title("Ultimate voice to text transformer ")
root.configure(bg=main_bg_color)

# main frame
main_frame = tk.Frame(root, bg=main_bg_color)
main_frame.pack(padx=20, pady=20)

# this is for  menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Options", menu=file_menu)
file_menu.add_command(label="Transcribe Audio", command=show_transcription_page)
file_menu.add_command(label="Summarize Text File", command=show_summary_page)
file_menu.add_command(label="Translate Text", command=show_translation_page)
file_menu.add_command(label="YouTube Transcription", command=show_youtube_transcription_page)


show_transcription_page()

root.mainloop()
