#  YT Summariser

YT Summariser is a YouTube video summarization app built using:

- **Streamlit** – for the frontend interface  
- **Whisper** – to transcribe video audio into text  
- **Haystack** – to handle prompt-based processing  
- **LLaMA 2** – as the language model for generating summaries  

Paste any YouTube link, and this app will fetch, transcribe, and summarize the video using powerful LLMs.

---

##  Setup Instructions

To set up and run the project, follow these steps:

### 1. Install the required packages

Create and activate a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```
Then install all dependencies:

```bash
pip install -r requirements.txt
```
### 2. Dowload the LLaMa 2 Model

Dowload the model using  : 
```bash
wget https://huggingface.co/TheBloke/Llama-2-7B-32K-Instruct-GGUF/resolve/main/llama-2-7b-32k-instruct.Q4_K_M.gguf
```

Make sure to dowload in the same directory as the yt_summary.py

### 3. Starting the application

```bash
streamlit run yt_summary.py
```

Processing might take a minute... good things (and summaries) take time! 😄