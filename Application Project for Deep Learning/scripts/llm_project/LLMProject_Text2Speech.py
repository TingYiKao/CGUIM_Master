from gtts import gTTS
from gtts.lang import tts_langs
from tempfile import NamedTemporaryFile

import ollama
import base64
import streamlit as st

langs = tts_langs().keys()

def autoplay_audio(file_path:str):
    with open(file_path, "rb") as file:
        data = file.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            ‹audio controls autoplay="true">
            ‹source src="data:audio/mp3;base64, {b64}" type="audio/mp3">
            Your browser does not support the audio element.
            </audio>
            """
        st.markdown(
            md, 
            unsafe_allow_html = True
        )

def main():
    st.title("LLM Project(Chatbot) M1244017 高定儀")
    st.write("This is a project for the Deep learning course")
    
    lang = st.selectbox("Select the language you want to use:", options = langs, index = 11)
    
    user_input = st.text_area("Please enter the text you want to generate:")
    
    if st.button("Generate"):
        if user_input:
            response = ollama.chat(model = 'mistral' ,messages = [{'role': 'user', 'content': user_input}])
            
            st.text('Answer:')
            st.write(response['message']['content'])
            
            tts = gTTS(response['message']['content'], lang = lang)
            
            gTTS (response ['message']['content'], lang=lang, slow  = False, lang_check = True)
            with NamedTemporaryFile(suffix = ".mp3", delete = False) as temp:
                tempname = temp.name
                tts.save(tempname)
                autoplay_audio(tempname)
        
        else:
            st.warning("Please enter the text you want to generate.")

if __name__ == "__main__":
    main()