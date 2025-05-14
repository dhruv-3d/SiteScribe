import streamlit as st
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from rag_pipeline import WebsiteScribber

# Initialize WebsiteScribber instance
if 'website_scribber' not in st.session_state:
    st.session_state['website_scribber'] = WebsiteScribber()

ws = st.session_state['website_scribber']

def main():
    st.title("Website Scribber Interface")

    # Sidebar for training
    with st.sidebar:
        st.header("Train on a Website")
        website_url = st.text_input("Enter website URL to train:")
        if st.button("Train"):
            if website_url:
                with st.spinner(f"Training on {website_url}..."):
                    try:
                        ws.train_on_website(website_url)
                        st.session_state['trained_url'] = website_url  # Store the trained URL
                        st.success(f"Successfully trained on: {website_url}")
                    except Exception as e:
                        st.error(f"Error during training: {e}")
            else:
                st.warning("Please enter a website URL.")

    # Main area for asking questions
    st.header("Ask Questions")
    user_question = st.text_input("Ask a question about the trained website:")
    if st.button("Get Response"):
        if user_question:
            if 'trained_url' in st.session_state and st.session_state['trained_url']:
                with st.spinner("Getting response..."):
                    try:
                        response = ws.ask_site_scribber(user_question)
                        st.subheader(f"Response for: {user_question}")
                        st.info(response)
                    except Exception as e:
                        st.error(f"Error getting response: {e}")
            else:
                st.warning("Please train on a website first before asking questions.")
        else:
            st.warning("Please enter your question.")

if __name__ == "__main__":
    main()