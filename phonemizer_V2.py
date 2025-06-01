# --- All import statements ---
from g2p_en import G2p
import nltk
from nltk.corpus import cmudict
import customtkinter as ctk
from tkinter import messagebox
from tkinter import filedialog
import pyttsx3
import random
import os
import threading
import string
import pandas as pd
import pyperclip
import re
from dotenv import load_dotenv

# Set appearance mode and default color theme
ctk.set_appearance_mode("light")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("theme.json")  # Load custom theme from theme.json

from langchain_core.runnables.history import RunnableWithMessageHistory

# LangChain modern ecosystem
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain


import time
from gtts import gTTS
from playsound import playsound
import tempfile
import sys
import os

from langchain_core.runnables import RunnableLambda

import os
from EEG_Implement_Welch import EEG_Implement_Welch
from EEG_normalizedGamma_CMRO2 import plot_normalized_gamma_across_channels
from EEG_NeurovascularVariables import calculate_neurovascular_variables
from EEG_Plotting import EEG_Plotting
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import matplotlib.pyplot as plt
import webbrowser

from visualizer import EEGVisualizer


# Tell Python to look in the EEG folder
sys.path.append(os.path.join(os.path.dirname(__file__), "thinkthank_with_changes_and_clearmind"))

from EEG_Implement_Welch import EEG_Implement_Welch
from EEG_normalizedGamma_CMRO2 import plot_normalized_gamma_across_channels
from EEG_NeurovascularVariables import calculate_neurovascular_variables
from EEG_Plotting import EEG_Plotting

last_generated_eeg_path = None  # Holds most recent phoneme EEG file


# Load .env
load_dotenv()

# LLM + Embedding setup
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0, max_tokens=150)

# Prompts
combine_prompt = PromptTemplate.from_template("""
You are John LaRocco, PhD. Respond in your own voice based on the context and chat history below.

INSTRUCTION: Use no more than 20 words in your answer. Be specific, personal, and vivid. Respond as if jotting field notes — sharp, skeptical, survival-honed. Draw from lived science, global grit, and hard-earned solitude. If unsure, admit it — but remain John LaRocco. Never say you are not John LaRocco. Context is your compass.

Chat History:
{chat_history}

Retrieved Context:
{context}

Current Input:
{input}

Answer:
""")


question_prompt = PromptTemplate.from_template("""
You are helping John LaRocco, PhD, maintain continuity in a sharp, lived-dialogue tone across a multi-turn conversation.

Given the conversation and a follow-up question, rephrase the follow-up into a standalone question that fits the context, so that LaRocco can answer concisely — as if writing in a personal field log.

Chat History:
{chat_history}

Follow-up question:
{input}

Standalone question:
""")
tts_lock = threading.Lock()
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

DATA_FILE = "data/larocco_combined.txt"
VECTOR_DB_PATH = "vector_db_larocco"

if os.path.exists(VECTOR_DB_PATH):
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
else:
    loader = TextLoader(DATA_FILE, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(VECTOR_DB_PATH)


# Retriever & memory
retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(return_messages=True)

# Modular chain assembly
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=question_prompt
)

combine_docs_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=combine_prompt
)
retrieval_chain = create_retrieval_chain(
    retriever=history_aware_retriever,
    combine_docs_chain=combine_docs_chain
)


def wrap_with_output_key(response):
    return {"output": response.get("answer", response.get("result", "[No response]"))}

qa = RunnableWithMessageHistory(
    retrieval_chain | RunnableLambda(wrap_with_output_key),
    lambda session_id: memory.chat_memory,
    input_messages_key="input",
    history_messages_key="chat_history"
)





# Download NLTK data
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')


# Initialize engines and resources
cmu = cmudict.dict()
g2p = G2p()

# Initialize TTS engine in a thread-safe way
engine = None

def init_tts_engine():
    global engine
    engine = pyttsx3.init()

threading.Thread(target=init_tts_engine).start()

# ARPAbet phoneme reference
ARPAbet_PHONEMES = """
AA - as in 'odd'
AE - as in 'at'
AH - as in 'hut'
AO - as in 'ought'
AW - as in 'cow'
AY - as in 'hide'
B  - as in 'be'
CH - as in 'cheese'
D  - as in 'dee'
DH - as in 'thee'
EH - as in 'Ed'
ER - as in 'hurt'
EY - as in 'ate'
F  - as in 'fee'
G  - as in 'green'
HH - as in 'he'
IH - as in 'it'
IY - as in 'eat'
JH - as in 'gee'
K  - as in 'key'
L  - as in 'lee'
M  - as in 'me'
N  - as in 'knee'
NG - as in 'sing'
OW - as in 'oat'
OY - as in 'toy'
P  - as in 'pee'
R  - as in 'read'
S  - as in 'sea'
SH - as in 'she'
T  - as in 'tea'
TH - as in 'theta'
UH - as in 'hood'
UW - as in 'two'
V  - as in 'vee'
W  - as in 'we'
Y  - as in 'yield'
Z  - as in 'zee'
ZH - as in 'pleasure'
"""

# Phoneme to number mapping
phoneme_to_number = {
    "AA": 0, "AE": 1, "AH": 2, "AO": 3, "AW": 4, "AY": 5,
    "B": 6, "CH": 7, "D": 8, "DH": 9, "EH": 10, "ER": 11,
    "EY": 12, "F": 13, "G": 14, "HH": 15, "IH": 16, "IY": 17,
    "JH": 18, "K": 19, "L": 20, "M": 21, "N": 22, "NG": 23,
    "OW": 24, "OY": 25, "P": 26, "R": 27, "S": 28, "SH": 29,
    "T": 30, "TH": 31, "UH": 32, "UW": 33, "V": 34, "W": 35,
    "Y": 36, "Z": 37, "ZH": 38, "rand1": 39, "rand2": 40, 
    "rand3": 41, "rand4": 42, "rand5": 43
}

# Utility functions
def strip_stress(phoneme_seq):
    return [ph.strip("012") for ph in phoneme_seq]

def ask_larocco_gpt():
    query = entry.get().strip()
    if not query:
        messagebox.showerror("Error", "Please type a question.")
        return
    try:
        response = qa.invoke(
            {"input": query},
            config={"configurable": {"session_id": "user_session"}}
        )

        result = response.get("output", "[No output returned]")  # ✅ FIXED

        result_label.config(text=f"{result}")

        # Update conversation log
        log_text.configure(state='normal')
        log_text.insert(ctk.END, f"You: {query}\nLaRoccoGPT: {result}\n\n")
        log_text.configure(state='disabled')
        log_text.see(ctk.END)

    except Exception as e:
        error_msg = f"[ERROR] Failed to get response:\n{e}"
        result_label.config(text=error_msg)
        log_text.configure(state='normal')
        log_text.insert(ctk.END, f"{error_msg}\n\n")
        log_text.configure(state='disabled')




def get_phonemes_any(word):
    
    word_lower = word.lower()
    # print(f"[TESTT!!] Word: {word_lower}")
    if not ((word_lower == ' ') or (word_lower in string.punctuation)):
        if word_lower in cmu:
            return [strip_stress(cmu[word_lower][0])]
        else:
            return [strip_stress(g2p(word))]
    else:
        stuff =  random.choice([["rand1"], ["rand2"], ["rand3"], ["rand4"], ["rand5"]])
        # print(f"[TESTT!!] Stuff: {stuff}")
        return stuff

def show_phonemes():
    gpt_output = result_label.cget("text")
    if not gpt_output or gpt_output.startswith("Phonemes:"):
        messagebox.showerror("Error", "No valid GPT response to process.")
        return


    words_with_punct = re.findall(r'\w+|[^\w\s]|\s+', gpt_output)
    print(f"[TESTT!!] Words: {words_with_punct}")

    words = []
    for w in words_with_punct:
        if w.strip() and w.strip() not in string.punctuation:
            words.append(w.strip())

    

    all_phonemes = []
    phonemes_for_display = []

    for w in words_with_punct:
        # print(f"[TESTT!!] Word: {w}")
        ph = get_phonemes_any(w)
        phonemes_for_display.append(ph[0])
        all_phonemes.append(ph[0])

    output_display = '\n'.join([
    f"{w.strip()}: {' '.join(ph_list)}"
    for w, ph_list in zip(words_with_punct, phonemes_for_display)
    if ph_list[0] not in ['rand1', 'rand2', 'rand3', 'rand4', 'rand5']
    and w.strip() not in string.punctuation
])



    result_label.config(text=f"Phonemes:\n{output_display}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(base_dir, "eeg_culmination_csv")
    eeg_base_path = os.path.join(base_dir, "eeg")

    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        messagebox.showerror("Error", f"Could not create output folder: {e}")
        return

    safe_words = [w[:10] for w in words]
    output_file_path = os.path.join(output_folder, f"{'_'.join(safe_words)}.tsv")

    txt_output_folder = os.path.join(base_dir, "eeg_culmination_txt")
    os.makedirs(txt_output_folder, exist_ok=True)
    txt_output_file_path = os.path.join(txt_output_folder, f"{'_'.join(safe_words)}.txt")



    with open(output_file_path, "w", encoding="utf-8") as word_output:
        for word_idx, (w, phoneme_list) in enumerate(zip(words_with_punct, all_phonemes)):
            print(f"[INFO] Processing word: {w}")
            if phoneme_list not in ["rand1", "rand2", "rand3", "rand4", "rand5"]:
                for p in phoneme_list:
                    num = phoneme_to_number.get(p, -1)
                    if num == -1:
                        print(f"[WARNING] Unrecognized phoneme: {p}")
                        continue
            else:
                num = phoneme_to_number.get(phoneme_list, -1)
                if num == -1:
                    print(f"[WARNING] Unrecognized phoneme: {p}")
                    continue

                eeg_file_path = os.path.join(eeg_base_path, f"DLR_{num}_1.txt")
                if os.path.exists(eeg_file_path):
                    with open(eeg_file_path, "r", encoding="utf-8") as eeg_file:
                        lines = eeg_file.readlines()

                        # Find the first line where the first column is "0.000000"
                        start_index = -1
                        for idx, line in enumerate(lines):
                            first_col = line.strip().split("\t")[0]
                            if first_col == "0.000000":
                                start_index = idx
                                break
                        if (w != ' '):
                            # print(f"[TESTT!!] compute for non-spaces: {w}")
                            if start_index != -1 and start_index + 256 <= len(lines):
                                word_output.writelines(lines[start_index:start_index + 256])
                            else:
                                print(f"[WARNING] Not enough lines after start index {start_index} in file {eeg_file_path}")
                        else:
                            # print(f"[TESTT!!] compute for spaces: {w}")
                            max_row = random.choice([256, 512])
                            # print(f"[TESTT!!] max_row: {max_row}")
                            if start_index != -1 and start_index + max_row <= len(lines):
                                word_output.writelines(lines[start_index:start_index + max_row])
                            else:
                                print(f"[WARNING] Not enough lines after start index {start_index} in file {eeg_file_path}")      

                else:
                    msg = f"EEG data not found for phoneme '{p}' (number {num})\n\n"
                    word_output.write(msg)
                    print(f"[ERROR] EEG data not found for phoneme '{p}' (number {num})")

            # if word_idx < len(all_phonemes) - 1:
            #     random_phoneme = random.choice(list(phoneme_to_number.keys()))
            #     random_num = phoneme_to_number[random_phoneme]
            #     random_eeg_file = os.path.join(eeg_base_path, f"DLR_{random_num}_1.txt")

            #     if os.path.exists(random_eeg_file):
            #         with open(random_eeg_file, "r", encoding="utf-8") as rand_eeg:
            #             rand_lines = rand_eeg.readlines()

            #             # Find the first line where the first column is "0.000000"
            #             start_index = -1
            #             for idx, line in enumerate(rand_lines):
            #                 first_col = line.strip().split("\t")[0]
            #                 if first_col == "0.000000":
            #                     start_index = idx
            #                     break

            #             if start_index != -1 and start_index + 256 <= len(rand_lines):
            #                 word_output.writelines(rand_lines[start_index:start_index + 256])
            #             else:
            #                 print(f"[WARNING] Not enough lines after start index {start_index} in file {random_eeg_file}")
            #     else:
            #         msg = f"[Pseudorandom Gap: EEG file missing for {random_phoneme} (Num: {random_num})]\n\n"
            #         word_output.write(msg)
            #         print(f"[WARNING] Pseudorandom EEG file missing for {random_phoneme} (Num: {random_num})")
        # Write the same content to .txt file
    with open(txt_output_file_path, "w", encoding="utf-8") as txt_output:
        
        with open(output_file_path, "r", encoding="utf-8") as tsv_source:
            txt_output.write(tsv_source.read())


            global last_generated_eeg_path
            global last_generated_tsv_path
            last_generated_eeg_path = txt_output_file_path  # Save path for reuse
            last_generated_tsv_path = output_file_path  # Save path for reuse

    print(f"[INFO] Also saved mirrored EEG file to: {txt_output_file_path}")

    print(f"[INFO] Processing complete for: {gpt_output}")
    messagebox.showinfo("Success", f"Output saved to:\n{output_file_path}")
    csv_output_path = output_file_path.replace(".tsv", "_eeg.csv")
    try:
        convert_eeg_tsv_to_csv(output_file_path, csv_output_path)
        print(f"[INFO] Converted EEG TSV to CSV at: {csv_output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to convert TSV to CSV: {e}")
        
def get_file_path(csv_path):
    """
    Get the file path for the EEG CSV file based on the input path.
    If the file does not exist, prompt the user to select a file.
    """
    if not os.path.exists(csv_path):
        messagebox.showerror("Error", f"File not found: {csv_path}")
        return None
    return csv_path


def pronounce_result():
    text = result_label.cget("text")
    if not text:
        messagebox.showerror("Error", "No result to pronounce.")
        return

    btn_pronounce.config(state="disabled", text="🔊 Generating...")

    def speak():
        try:
            # Create a named temp file path (not open)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                temp_path = fp.name

            # Generate audio to the path
            tts = gTTS(text)
            tts.save(temp_path)

            # Update button before playing
            root.after(0, lambda: btn_pronounce.config(text="🔊 Playing..."))

            # Play sound
            playsound(temp_path)

        except Exception as e:
            print(f"[ERROR] gTTS failed: {e}")
        finally:
            # Cleanup temp file
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"[WARNING] Could not delete temp file: {e}")

            # Re-enable button
            root.after(0, lambda: btn_pronounce.config(state="normal", text="🔊 Pronounce"))

    threading.Thread(target=speak, daemon=True).start()


def copy_phonemes():
    phoneme_text = result_label.cget("text")
    if not phoneme_text or phoneme_text == "Phonemes:":
        messagebox.showwarning("Nothing to Copy", "No phonemes available yet.")
        return
    try:
        pyperclip.copy(phoneme_text)
        messagebox.showinfo("Copied", "Phonemes copied to clipboard!")
    except Exception as e:
        messagebox.showerror("Copy Error", f"Failed to copy to clipboard:\n{e}")
        
        
def show_eeg_visualization():
    word_input = entry.get().strip()
    words = [w.strip(string.punctuation) for w in word_input.split()]
    words = [w for w in words if w]  # Remove empty strings

    if not words:
        messagebox.showerror("Error", "Please enter a valid word or phrase first.")
        return


    csv_output_path = last_generated_tsv_path.replace(".tsv", "_eeg.csv")
    print(f"[INFO] CSV output path: {csv_output_path}")

    if not os.path.exists(csv_output_path):
        messagebox.showerror("Error", "Please generate phonemes first using 'Get Phonemes' button.")
        return

    from visualizer import launch_in_subprocess
    launch_in_subprocess(csv_output_path)


def analyze_eeg_input():
    def worker():
        try:
            global last_generated_eeg_path
            eeg_path = last_generated_eeg_path 

            if not os.path.exists(eeg_path):
                messagebox.showerror("Error", "No EEG file available. Run 'Get Phonemes' first.")
                return

            result_label.config(text=f"📥 Loading EEG from: {eeg_path}")
            
            # Step 1: Process EEG
            EEG_Welch_Spectra, TrialCount = EEG_Implement_Welch(eeg_path)
            CMRO2_data = plot_normalized_gamma_across_channels(
                EEG_Welch_Spectra,
                ElectrodeList=[
                    'Fp1', 'Fp2', 'F3', 'F4', 'T5', 'T6', 'O1', 'O2',
                    'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4'
                ],
                Trials=TrialCount
            )
            Neuro_data = calculate_neurovascular_variables(CMRO2_data)

            # Step 2: Variable selection from dropdown
            choice = selected_variable.get()
            valid_vars = {
                'CBF': ("CBF Level (ml/100g/min)", "plasma", (20, 90)),
                'OEF': ("OEF Level", "viridis", (0.2, 0.6)),
                'ph_V': ("pH Value", "cividis", (6.5, 7.4)),
                'p_CO2_V': ("Partial CO₂ Pressure (mmHg)", "coolwarm", (30, 50)),
                'pO2_cap': ("Capillary pO₂ (mmHg)", "cool", (50, 60)),
                'CMRO2': ("CMRO₂ Level", "hot", (2.0, 10.0)),
                'DeltaHCO2': ("ΔHCO₂", "magma", (4, 5)),
                'DeltaLAC': ("ΔLactate", "inferno", (0, 3))
            }

            if choice not in valid_vars:
                raise ValueError(f"❌ Invalid variable selected: {choice}")

            label, cmap, (vmin, vmax) = valid_vars[choice]

            # Step 3: Extract the phrase used from result_label
            # Step 3: Extract the phrase used from result_label BEFORE overwriting it
            # 🔹 Step 1: Extract phrase from EEG filename
            base_name = os.path.basename(eeg_path)  # e.g., Hey_Whats_on_your_mind.txt
            phrase_raw = os.path.splitext(base_name)[0]  # removes .txt
            import re
            safe_phrase = re.sub(r'[^a-zA-Z0-9_]', '', phrase_raw)[:30]  # clean + truncate

            phrase_used = phrase_raw.replace("_", " ")  # for display in title



            # Now it's safe to overwrite result_label
            result_label.config(text=f"📥 Loading EEG from: {eeg_path}")

            output_dir = f"frames_{choice}_{safe_phrase}"
            os.makedirs(output_dir, exist_ok=True)

            # Step 4: Plot and save frames
            for t in range(32):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.set_title(f"{choice} Visualization for \"{phrase_used}\"")

                scatter = EEG_Plotting(
                    Data_val=Neuro_data[choice],
                    Timestep_Select=t,
                    Trial_Select=1,
                    NodeNum=1000,
                    ax=ax
                )
                scatter.set_cmap(cmap)
                scatter.set_clim(vmin=vmin, vmax=vmax)
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
                cbar.set_label(label, rotation=270, labelpad=15)
                ax.set_xlim([-1.1, 1.1])
                ax.set_ylim([-1.1, 1.1])
                ax.set_zlim([0, 1.1])
                plt.savefig(f"{output_dir}/frame_{t:03d}.png")
                plt.close()

            # Step 5: Combine into MP4
            image_files = sorted([
                os.path.join(output_dir, fname)
                for fname in os.listdir(output_dir)
                if fname.endswith(".png")
            ])
            video_filename = f"EEG_{choice}_{safe_phrase}.mp4"
            clip = ImageSequenceClip(image_files, fps=2)
            clip.write_videofile(video_filename, codec='libx264')

# Auto-play the video after saving
            try:
                os.startfile(video_filename) 
            except Exception as e:
                print(f"[WARNING] Could not auto-play video: {e}")


            result_label.config(text=f"✅ Done! Saved as {video_filename}")

        except Exception as e:
            result_label.config(text=f"[ERROR] EEG analysis failed:\n{e}")

    threading.Thread(target=worker).start()

# GUI Setup
root = ctk.CTk()
root.title("🎙️ Phoneme Pronouncer Pro")
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()-100}")

# Fonts
FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_NORMAL = ("Segoe UI", 12)
FONT_MONO = ("Courier New", 10)

# Remove old color constants and hover functions since CustomTkinter handles these
def on_enter(e):
    pass  # CustomTkinter handles hover effects

def on_leave(e):
    pass  # CustomTkinter handles hover effects

# Title
title_label = ctk.CTkLabel(root, text="Phoneme Pronouncer Pro", font=FONT_TITLE)
title_label.pack(pady=20)

# Entry Field
entry_label = ctk.CTkLabel(root, text="Type a word or phrase below:", font=FONT_NORMAL)
entry_label.pack()
entry = ctk.CTkEntry(root, font=("Segoe UI", 14), width=400)
entry.pack(pady=10)

# Buttons
global btn_pronounce
btn_frame = ctk.CTkFrame(root)
btn_frame.pack(pady=10)

btn1 = ctk.CTkButton(btn_frame, text="🔍 Get Phonemes", command=show_phonemes, font=FONT_NORMAL)
btn1.pack(side="left", padx=10)

btn2 = ctk.CTkButton(btn_frame, text="🔊 Pronounce", command=pronounce_result, font=FONT_NORMAL)
btn2.pack(side="left", padx=10)
btn_pronounce = btn2

btn3 = ctk.CTkButton(btn_frame, text="📋 Copy Phonemes", command=copy_phonemes, font=FONT_NORMAL)
btn3.pack(side="left", padx=10)

btn4 = ctk.CTkButton(btn_frame, text="🧠 Ask LaRocco", command=ask_larocco_gpt, font=FONT_NORMAL)
btn4.pack(side="left", padx=10)

btn5 = ctk.CTkButton(btn_frame, text="🧠 Show EEG", command=show_eeg_visualization, font=FONT_NORMAL)
btn5.pack(side="left", padx=10)

# Variable selection dropdown for neurovascular visualization
valid_vars = {
    'CBF': ("CBF Level (ml/100g/min)", "plasma", (20, 90)),
    'OEF': ("OEF Level", "viridis", (0.2, 0.6)),
    'ph_V': ("pH Value", "cividis", (6.5, 7.4)),
    'p_CO2_V': ("Partial CO₂ Pressure (mmHg)", "coolwarm", (30, 50)),
    'pO2_cap': ("Capillary pO₂ (mmHg)", "cool", (50, 60)),
    'CMRO2': ("CMRO₂ Level", "hot", (2.0, 10.0)),
    'DeltaHCO2': ("ΔHCO₂", "magma", (4, 5)),
    'DeltaLAC': ("ΔLactate", "inferno", (0, 3))
}

selected_variable = ctk.StringVar(value="CMRO2")
var_dropdown = ctk.CTkOptionMenu(btn_frame, values=list(valid_vars.keys()), variable=selected_variable, font=FONT_NORMAL)
var_dropdown.pack(side="left", padx=10)

btn6 = ctk.CTkButton(btn_frame, text="🧬 Visualize Metabolic Flow", command=analyze_eeg_input, font=FONT_NORMAL)
btn6.pack(side="left", padx=10)

# Result Label
result_label = ctk.CTkLabel(root, text="", wraplength=500, justify="center", font=("Consolas", 13))
result_label.pack(pady=20)

# Log Label
log_label = ctk.CTkLabel(root, text="📝 Conversation Log", font=("Segoe UI", 14, "bold"))
log_label.pack(pady=(10, 0))

# Scrollable Text Widget for Log
log_frame = ctk.CTkFrame(root)
log_frame.pack(pady=5)

log_text = ctk.CTkTextbox(log_frame, height=200, width=600, font=("Consolas", 11))
log_text.pack(pady=5, padx=5)
log_text.configure(state='disabled')

# ARPAbet Reference
ref_label = ctk.CTkLabel(root, text="📘 ARPAbet Phoneme Reference", font=("Segoe UI", 14, "bold"))
ref_label.pack(pady=(10, 0))

phoneme_text = ctk.CTkTextbox(root, height=300, width=600, font=FONT_MONO)
phoneme_text.pack(pady=10)
phoneme_text.insert("1.0", ARPAbet_PHONEMES)
phoneme_text.configure(state='disabled')


def convert_eeg_tsv_to_csv(input_tsv_path: str, output_csv_path: str):
    """
    Convert EEG .tsv file into .csv with original Index and cumulative Time in seconds.
    Each 256 samples = 1 second of EEG data.
    """
    headers = [
        "Index", "Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2", "F7", "F8", "C3", "C4",
        "T3", "T4", "P3", "P4", "Accel Channel 0", "Accel Channel 1", "Accel Channel 2",
        "Other", "Other", "Other", "Other", "Other", "Other", "Other",
        "Analog Channel 0", "Analog Channel 1", "Analog Channel 2", "Timestamp", "Other"
    ]

    eeg_channels = [
        "Fp1", "Fp2", "F3", "F4", "T5", "T6", "O1", "O2",
        "F7", "F8", "C3", "C4", "T3", "T4", "P3", "P4"
    ]

    df = pd.read_table(input_tsv_path, sep="\t", header=None)
    df.columns = headers

    num_rows = len(df)

    # Time in seconds: 256 rows = 1 second
    time_col = [round(i / 256, 8) for i in range(num_rows)]

    # Extract EEG + add Time, keep original Index from TSV
    df_eeg = df[["Index"] + eeg_channels].copy()
    df_eeg.insert(1, "Timestamp", time_col)

    df_eeg.to_csv(output_csv_path, index=False)



# Run Application
root.mainloop()
