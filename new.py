from langchain_community.llms import Ollama
import streamlit as st
from streamlit import session_state
import time
import base64
import os
from vectors import EmbeddingsManager  # Import the EmbeddingsManager class
from chatbot import ChatbotManager     # Import the ChatbotManager class
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import time

# Ensure Ollama is correctly imported. Adjust the import path if needed.
from langchain.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

# Function to display the PDF of a given file
def displayPDF(file):
    # Reading the uploaded file
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying the PDF
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize session_state variables if not already present
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Set the page configuration to wide layout and add a title
st.set_page_config(
    page_title="Insuratron by Keshav Singh",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    # You can replace the URL below with your own logo URL or local image path
    st.image("logo1.jpg", use_column_width=True)
    st.markdown("### 📚 Your Personal Insurance document assistant")
    st.markdown("---")
    
    # Navigation Menu
    menu = ["🏠 Home", "🤖 Insuratron", "🤖LLAMA chatbot"]
    choice = st.selectbox("Navigate", menu)

# Home Page
if choice == "🏠 Home":
    st.title("📄 INSURATRON By Keshav Singh 🚀")
    st.markdown("""
    Welcome this project is Gen AI RAG based project,

    Built using Open Source Stack (Llama 3.2, BGE Embeddings, and Qdrant running locally within a Docker Container.)

    - Upload Documents: Easily upload your PDF documents.
    - Summarize: Get concise summaries of your documents.
    - Chat: Interact with your documents through our intelligent chatbot.
    - Enhancements in working**: Further trying to implement the base model as the 
bitext
/
Mistral-7B-Insurance.(Requires more computational resources and power
further i believe the power of Generator and Discriminator networks could be utilised to enhance the model finetuning)
                
   -- **if you want to enhance the embedding believing you have computational access please go ahead and try it out by replacing the configurations in the embeeding module with the required module. 
                https://huggingface.co/FinLang/finance-embeddings-investopedia 
    Enhance your document management experience with Insuratron! 😊
    """)

# Chatbot Page
elif choice == "🤖 Insuratron":
    st.markdown("---")
    
    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Column 1: File Uploader and Preview
    with col1:
        st.header("📂 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            st.success("📄 File Uploaded Successfully!")
            # Display file name and size
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**File Size:** {uploaded_file.size} bytes")
            
            # Display PDF preview using displayPDF function
            st.markdown("### 📖 PDF Preview")
            displayPDF(uploaded_file)
            
            # Save the uploaded file to a temporary location
            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Store the temp_pdf_path in session_state
            st.session_state['temp_pdf_path'] = temp_pdf_path

    # Column 2: Create Embeddings
    with col2:
        st.header("🧠 Embeddings")
        create_embeddings = st.checkbox("✅ Create Embeddings")
        if create_embeddings:
            if st.session_state['temp_pdf_path'] is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                try:
                    # Initialize the EmbeddingsManager  ## in future try different domain specific model
                    embeddings_manager = EmbeddingsManager(   
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        qdrant_url="http://localhost:6333",
                        collection_name="vector_db"
                    )
                    
                    with st.spinner("🔄 Embeddings are in process..."):
                        # Create embeddings
                        result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                        time.sleep(1)  # Optional: To show spinner for a bit longer
                    st.success(result)
                    
                    # Initialize the ChatbotManager after embeddings are created
                    if st.session_state['chatbot_manager'] is None:
                        st.session_state['chatbot_manager'] = ChatbotManager(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            llm_model="llama3.2:3b",
                            llm_temperature=0.7,
                            qdrant_url="http://localhost:6333",
                            collection_name="vector_db"
                        )
                    
                except FileNotFoundError as fnf_error:
                    st.error(fnf_error)
                except ValueError as val_error:
                    st.error(val_error)
                except ConnectionError as conn_error:
                    st.error(conn_error)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # Column 3: Chatbot Interface
    with col3:
        st.header("💬 Chat with Document")
        
        if st.session_state['chatbot_manager'] is None:
            st.info("🤖 Please upload a PDF and create embeddings to start chatting.")
        else:
            # Display existing messages
            for msg in st.session_state['messages']:
                st.chat_message(msg['role']).markdown(msg['content'])

            # User input
            if user_input := st.chat_input("Type your message here..."):
                # Display user message
                st.chat_message("user").markdown(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})

                with st.spinner("🤖 Responding..."):
                    try:
                        # Get the chatbot response using the ChatbotManager
                        answer = st.session_state['chatbot_manager'].get_response(user_input)
                        time.sleep(1)  # Simulate processing time
                    except Exception as e:
                        answer = f"⚠️ An error occurred while processing your request: {e}"
                
                # Display chatbot message
                st.chat_message("assistant").markdown(answer)
                st.session_state['messages'].append({"role": "assistant", "content": answer})
elif choice == "🤖LLAMA chatbot":
# Initialize session state for messages if it doesn't already exist
    if 'llama_messages' not in st.session_state:
        st.session_state['llama_messages'] = []

    # LLaMA Chatbot section
    st.title("🤖 LLaMA Model Only Chatbot (Plain LLaMA 🥙)")
    st.markdown("---")

    # Display existing messages for chatbot
    st.header("💬 Chat with LLaMA Model")
    if len(st.session_state['llama_messages']) == 0:
        st.write("No messages yet. Start the conversation!")

    for msg in st.session_state['llama_messages']:
        st.chat_message(msg['role']).markdown(msg['content'])

    # User input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # Display user's message and add it to session state
        st.chat_message("user").markdown(user_input)
        st.session_state['llama_messages'].append({"role": "user", "content": user_input})

        # Generate a response from the Ollama LLaMA model
        with st.spinner(f"🤖 LLaMA Model is Responding..."):
            try:
                # Initialize and get response from LLaMA model using Ollama
                llama_model = Ollama(model='llama3.2:3b')
                model_answer = llama_model(user_input)  # Call the model to generate the response
            except Exception as e:
                model_answer = f"⚠️ An error occurred while processing your request: {e}"

            time.sleep(1)  # Simulate processing time

        # Display LLaMA's response and add it to session state
        st.chat_message("assistant").markdown(model_answer)
        st.session_state['llama_messages'].append({"role": "assistant", "content": model_answer})
