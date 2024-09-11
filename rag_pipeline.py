# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
os.environ['USER_AGENT'] = 'RagApp/1.0'
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import llm, embeddings_model


class WebsiteScribber():
    
    def __init__(self):
        self.website_url = None
        self.vectorstore = None
        self.site_scribber = None
        self.list_of_links = None


    def scrape_links_from_website(self):
        try:
            print(f"Searching [{self.website_url}] for more related URLs...")
            response = requests.get(self.website_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
        except requests.exceptions.RequestException as err:
            print(f"Error fetching data: {err}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        links = []

        links.append(self.website_url)
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href not in links:
                if href.startswith('/'):
                    links.append(f"{self.website_url}{href}")
                else:
                    links.append(href)

        filtered_links = []
        # Filter out specific patterns
        for link in links:
            if not (link.startswith('#') or link.startswith('tel:') or '/blog' in link or
                    link.endswith('.jpg') or link.endswith('.png') or 'mailto:' in link or
                    link.endswith('.gif') or link.endswith('.jpeg')):
                filtered_links.append(link)


        print(f"\nWebsite: {self.website_url}\nFollowing links were found within it: \n{filtered_links}")

        return filtered_links


    def load_webpages(self):
        loader = WebBaseLoader(web_paths=self.list_of_links)
        docs = loader.load()
        print(docs[0])

        print(f"Splitting page contents into smaller pieces for meaningful storage and retrieval...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        return splits


    def embed_and_store_docs(self):
        print(f"Extracting page contents from the website [{self.website_url}] ...")
        docs_to_store = self.load_webpages()

        print(f"Converting website contents into to vectors and storing in vector DB.\nThis could take few minutes, please wait...")
        self.vectorstore = Chroma.from_documents(documents=docs_to_store, embedding=embeddings_model)
        print(f"Website content stored and ready for use!")

    
    def setup_rag_pipeline(self):

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = self.vectorstore.as_retriever()

        rag_prompt_template = """You are an AI assistant specialising in information retrieval and analysis. Answer the following question based only on the given context:

Context: {context}

Question: {question}"""
        prompt = PromptTemplate.from_template(rag_prompt_template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.site_scribber = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def train_on_website(self, website_url):
        self.website_url = website_url
        self.list_of_links = self.scrape_links_from_website()

        print(f"Initiating training for website: [{self.website_url}]...")
        self.embed_and_store_docs()

        print(f"Almost done, finalizing things...")
        self.setup_rag_pipeline()
        print(f"Done! Training completed! You can go ahead and ask questions about your website [{self.website_url}]")


    def ask_site_scribber(self, user_query):
        response = self.site_scribber.invoke(user_query)
        return response
    

if __name__ == '__main__':
    ws = WebsiteScribber('https://www.zestratech.com')
    ws.train_on_website()

    resp = ws.ask_site_scribber("Do you provide mobile application development service? Which technology do you use for mobile apps?")
    print(resp)