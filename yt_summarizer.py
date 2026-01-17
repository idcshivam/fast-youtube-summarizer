import os
import streamlit as st
import whisper
import yt_dlp
from transformers import pipeline
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from textblob import TextBlob
from fpdf import FPDF
from gtts import gTTS

# Load lightweight models (faster)
model = whisper.load_model("tiny")
summarizer = pipeline("summarization", model="t5-small")

# Use environment variable for API key
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "YOUR_API_KEY_HERE")

def download_youtube_audio(url, max_duration=600):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            title = info_dict.get('title', 'No Title')
            thumbnail_url = info_dict.get('thumbnail', '')
            video_id = info_dict.get('id', '')

            if info_dict.get("duration", 0) > max_duration:
                st.warning(f"⏩ Video is long. Only first {max_duration//60} minutes will be processed.")
            
            return "audio.mp3", title, thumbnail_url, video_id
    except Exception as e:
        st.error(f"❌ Error downloading video: {e}")
        return None, None, None, None

@st.cache_data
def transcribe_audio(file_path):
    try:
        result = model.transcribe(file_path)
        return result['text']
    except Exception as e:
        st.error(f"❌ Transcription error: {e}")
        return ""

def summarize_text(text, chunk_size=300):
    if not text:
        return "No text to summarize"
    
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summary = ""
    
    for chunk in chunks:
        if len(chunk.strip()) < 20:
            continue
        try:
            summary_piece = summarizer(
                chunk,
                max_length=60,
                min_length=20,
                do_sample=False
            )[0]['summary_text']
            summary += summary_piece + " "
        except Exception:
            summary += "[Skipped Chunk] "
    
    return summary.strip() if summary else "Could not generate summary"

def get_comments_sentiment(video_id, max_results=5):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        request = youtube.commentThreads().list(
            part="snippet", 
            videoId=video_id, 
            maxResults=max_results,
            textFormat="plainText"
        )
        response = request.execute()

        sentiments = []
        for item in response.get('items', []):
            try:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                sentiment = TextBlob(comment).sentiment.polarity
                sentiments.append((comment, sentiment))
            except Exception:
                continue
        
        return sentiments
    except HttpError as e:
        st.warning(f"⚠️ Could not fetch comments: {e}")
        return []

def create_pdf(text, filename):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=11)
        pdf.set_auto_page_break(auto=True, margin=15)
        
        for line in text.split('\n'):
            if line.strip():
                pdf.multi_cell(0, 10, line)
        
        pdf.output(filename)
    except Exception as e:
        st.error(f"❌ PDF creation error: {e}")

def generate_audio(text, filename):
    try:
        if text and len(text) > 10:
            tts = gTTS(text, lang='en', slow=False)
            tts.save(filename)
    except Exception as e:
        st.error(f"❌ Audio generation error: {e}")

st.set_page_config(page_title="YouTube Summarizer", layout="wide")
st.title("⚡ Fast YouTube Summarizer")

youtube_url = st.text_input("🔗 Enter YouTube Video URL:")

if youtube_url:
    with st.spinner("⏳ Processing video... Please wait"):
        try:
            audio_file, title, thumbnail, video_id = download_youtube_audio(youtube_url)
            
            if not audio_file:
                st.stop()
            
            transcript = transcribe_audio(audio_file)
            summary = summarize_text(transcript)
            
            create_pdf(transcript, "Transcript.pdf")
            generate_audio(summary, "summary.mp3")
            
            # Clean up audio file
            if os.path.exists(audio_file):
                os.remove(audio_file)

            col1, col2 = st.columns(2)
            
            with col1:
                if thumbnail:
                    st.image(thumbnail, use_column_width=True)
            
            with col2:
                st.subheader("🎬 Title:")
                st.write(title)

            st.subheader("📝 Summary:")
            st.write(summary)

            if os.path.exists("summary.mp3"):
                st.audio("summary.mp3")

            with st.expander("📜 Full Transcript"):
                st.write(transcript)

            if os.path.exists("Transcript.pdf"):
                with open("Transcript.pdf", "rb") as file:
                    st.download_button("📥 Download Transcript PDF", file, file_name="Transcript.pdf")

            comments = get_comments_sentiment(video_id)
            
            if comments:
                st.subheader("💬 Top Comments Sentiment")
                for comment, sentiment in comments:
                    sentiment_text = "Positive 😊" if sentiment > 0.1 else "Negative 😠" if sentiment < -0.1 else "Neutral 😐"
                    st.write(f"> {comment[:200]}...")
                    st.caption(f"Sentiment: {sentiment_text} ({sentiment:.2f})")
            else:
                st.info("ℹ️ No comments found or comments disabled")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")