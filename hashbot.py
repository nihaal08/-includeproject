import streamlit as st
import google.generativeai as genai

genai.configure(api_key="AIzaSyABpTOzU6vT8jEljjMRTpbKF3oTSnm7tJg")  
model = genai.GenerativeModel("gemini-1.5-flash")

st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
    }

    .title {
        font-size: 2.2em;
        font-weight: 600;
        color: brown; 
        text-align: center;
        margin-top: 20px;
    }

    .subtitle {
        font-size: 20px;
        text-align: center;
        padding-bottom: 50px;
        color: #D3D3D3; 
    }

    .instructions {
        font-size: 1em;
        color: #A9A9A9; 
        text-align: center;
        margin-bottom: 20px;
    }

    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }

    .stButton > button {
        background-color: #F5F5DC; 
        color: #4B0082; 
        justify-content: center;
        font-size: 1em;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 6px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #8B4513; 
        color: #FFFAF0; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>HashBot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Where Your Questions Meet Intelligent Responses</div>", unsafe_allow_html=True)

user_input = st.text_input("Enter your question here:")
st.markdown("<div class='instructions'>Please press Enter before clicking 'Generate Response'</div>", unsafe_allow_html=True)

st.markdown("<div class='button-container'>", unsafe_allow_html=True)
if st.button("Generate Response"):
    if user_input:
        with st.spinner('Generating response...'):
            try:
                response = model.generate_content(user_input)
                st.write("### Response")
                st.write(response.text)
            except Exception as e:
                error_message = str(e)
                if "404" in error_message:
                    st.error("Error: The model could not be found. Please check the model name or your API key.")
                elif "500" in error_message:
                    st.error("Server Error: There might be an issue with the API service. Please try again later.")
                else:
                    st.error("An error occurred: Please check your input and try again.")
    else:
        st.warning("Please enter a question to get a response.")
st.markdown("</div>", unsafe_allow_html=True)
