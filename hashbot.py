import streamlit as st
import google.generativeai as genai

# Configure your Google Generative AI API key
genai.configure(api_key="AIzaSyABpTOzU6vT8jEljjMRTpbKF3oTSnm7tJg")  # Replace with your actual API key

# Initialize the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit app layout
st.markdown(
    """
    <style>
    body {
        font-family: Arial, sans-serif;
    }

    .title {
        color:brown;
        font-size: 2.2em;
        font-weight: 600;
        color: #F5F5DC; 
        text-align: center;
        margin-top: 20px;
    }

    .title1 {
        font-size: 20px;
        text-align: center;
        padding-bottom: 50px;
    }
    
    /* Instruction text styling */
    .instructions {
        font-size: 1em;
        color: brown;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Center alignment for button container */
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }

    .stButton>button {
        background-color: #F5F5DC;
        color: #ffffff;
        justify-content: center;
        font-size: 1em;
        font-weight: 600;
        padding: 10px 24px;
        border-radius: 6px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: brown;
        color: #F5F5DC;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>HashBot</div>", unsafe_allow_html=True)
st.markdown("<div class='title1'>Where Your Questions Meet Intelligent Responses</div>", unsafe_allow_html=True)

# Input box for user to enter their question
user_input = st.text_input("Enter your question here:")
st.markdown("<div class='instructions'>Please press Enter before clicking 'Generate Response'</div>", unsafe_allow_html=True)

# Center-align the button
st.markdown("<div class='button-container'>", unsafe_allow_html=True)
if st.button("Generate Response"):
    if user_input:
        try:
            # Display response as plain text without CSS
            response = model.generate_content(user_input)
            st.write("### Response")
            st.write(response.text)
        except Exception as e:
            st.error("An error occurred while generating the response.")
            st.write(e)
    else:
        st.warning("Please enter a question to get a response.")
st.markdown("</div>", unsafe_allow_html=True)
