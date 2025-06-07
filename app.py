import streamlit as st
from pipeline import process_video_url

st.title("🎙️ English Accent Classifier Tool")

st.write("""
Provide a public video URL (YouTube, Loom, MP4 link, etc.).
The tool will:
- Extract the audio track
- Check if the speech is in English
- Classify the English accent (British, American, Australian, etc.)
- Return a confidence score
""")

video_url = st.text_input("Enter Video URL")

if st.button("Run Accent Analysis"):
    with st.spinner("Processing... this may take a minute."):
        try:
            accent_label, confidence_score, result_summary = process_video_url(video_url)
            st.success(f"Predicted Accent: {accent_label}")
            st.info(f"Confidence Score: {confidence_score:.2f}%")
            st.write(result_summary)
        except Exception as error:
            st.error(f"An error occurred: {str(error)}")

