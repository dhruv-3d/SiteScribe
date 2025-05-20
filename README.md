# SiteScribe
Parses the given website and users can then have a chat with their website.

## Brief Description:
SiteScribe does the scraping for website links, loads and splits the content of all the links it can find, embedds it into a vector database, and uses a RAG (Retrieval-Augmented Generation) pipeline to answer queries based on website content.

## Installation:
#### Install Python dependencies:
> pip install -r requirements.txt

#### Expose & run backend APIs:
> fastapi run app.py

#### Spin up a quick demo playground:
> streamlit run ui.py


Please Note:
* It's still a work in progress, but core backend logic is done.
* This demo is powered by Google AI Studio, you will need an API key from Google AI Studio
* Checkout rag_pipeline.py to try it out yourself
