import ollama
import streamlit as st

def main():
    st.title("LLM Project M1244017 高定儀")
    st.write("This is a project for the Deep learning course")
    
    user_input = st.text_area("Please enter the text you want to generate:")
    
    if st.button("Generate"):
        if user_input:
            response = ollama.chat(model = 'mistral' ,messages = [{'role': 'user', 'content': user_input}])
            
            st.text('Answer:')
            st.write(response['message']['content'])
        else:
            st.warning("Please enter the text you want to generate.")

if __name__ == "__main__":
    main()