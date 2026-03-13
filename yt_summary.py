import streamlit as st  #for the UI
from haystack.nodes import PromptNode, PromptModel  #PromtModel bascially sets up lower level of LLM like the hardware and the no.of tokens and all 
#PromtNode it guides the started LLM what to do Haystack has some templates like for summarisation or Q/A and we use those templates to guide the LLM
from haystack.nodes.audio import WhisperTranscriber #this helps in converting the audio file to text(from OpenAI)
from haystack.pipelines import Pipeline
# The Pipeline helps organize and connect multiple components in a flow (e.g., first transcribe audio, then summarize text).
from model_add import LlamaCPPInvocationLayer
# Custom class that lets Haystack use llama-cpp-python models efficiently on local hardware.
# This adjusts settings like token count, memory mapping, number of threads, etc.
import time #to calucalte the runtime
import yt_dlp #to dowload audio from YT link that we give

st.set_page_config(
    layout="wide",
    page_title='YT Summariser'

)

#can also use pytube also instead of yt_dlp
def download_video(url):
    ydl_opts = {   #to select options
        'format': 'bestaudio/best',#if we use bestvideo we can even dowload video
        'outtmpl': 'audio.%(ext)s',#naming our output file as audio and the format can be any as it receives
        'quiet': True,#this hides all the logs that happen while dowloading
        'postprocessors': [{    #to do changes to the dowloaded file
            'key': 'FFmpegExtractAudio',#we use FFmpegExtractAudio to extract the audio
            'preferredcodec': 'mp3',#we force to get the audio in mp3 format
            'preferredquality': '192',#the quality 192kbps is good in file size and quality
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:#this specifies where we use these options and fetch the data from 
        info = ydl.extract_info(url, download=True)#we extract the things using the defined options and dowload true means we will dowload the file along with meta info also
        return f"audio.mp3"#returns the path of the audio file

def initialize_model(full_path):#we use PromptModel to initialize the LLM here we are using LLaMa 2.0
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,#the one which informs the format we want to use the LLM as
        use_gpu=False,#we use only CPU restricting GPU being used
        max_length=512 #we limit the tokens to 512 (tokens are the bunch of data the LLM reads info as LLama reads as token is ~3/ of english word)
        #LLMs have a maximum token limit (like 4096, 8192, 32,000 depending on the model)
    )

def initialize_prompt_node(model):#we guide the LLM like say it what to do we use PromptNode which has predefined template to summarise in Haystack
    summary_prompt = "deepset/summarization"#the template name is given
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)#we send that template to required model

def transcribe_audio(file_path, prompt_node):#this works in converting audio to text
    whisper = WhisperTranscriber()#initializing the transcriber
    pipeline = Pipeline()#initializing the Pipeline to carry steps of the process
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])#step 1 of pipeline is to recieve an input file and transcribe it using whisper
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])#step2 is using the output of whisper as input we will get the summary using the prompt we have
    #name here is the node name we given and the component is the one which does the work 
    output = pipeline.run(file_paths=[file_path])#we start the pipeline where it starts by running step1 and so on
    return output#the final output is returned

def main():
    st.title("YouTube Video Summarizer")
    st.markdown('<style>h1{color: white; text-align: center;}</style>', unsafe_allow_html=True)#edit the styling accordingly
    with st.expander("About the App"): #for about the app
        st.write("Bored of watching a YT video? Lets summarise it then!")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start.")
    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")
    # Submit button
    if st.button("Submit") and youtube_url: #if button is clicked and URL is there
        with st.spinner("Processing video..."):
            start_time = time.time()  # Start the timer
            # Download video
            file_path = download_video(youtube_url)
            # Initialize model
            full_path = "llama-2-7b-32k-instruct.Q4_K_M.gguf"
            model = initialize_model(full_path)
            prompt_node = prompt_node = initialize_prompt_node(model)
            # Transcribe audio
            output = transcribe_audio(file_path, prompt_node)
            end_time = time.time()  # End the timer
            elapsed_time = end_time - start_time
        # Display output with 2 columns
        col1, col2 = st.columns([1,1])
        # Column 1: Video view
        with col1:
            st.video(youtube_url)
        # Column 2: Summary View
        # Extract and display cleaned summary
        summary = output["results"][0].split("\n\n[INST]")[0]
        #This splits the string at the marker \n\n[INST], which is often used internally by LLMs like LLaMA 2 to separate the prompt/instruction from the response or metadata.
        with col2:
            st.markdown("### 📝 Summary")
            st.success(summary)
            st.markdown(f"⏱️ **Time taken**: `{elapsed_time:.2f}` seconds")

if __name__ == "__main__":
    main()