# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
import requests
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_transformers import MarkdownifyTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import llm, embeddings_model


class WebsiteScribber():
    
    def __init__(self):
        self.website_url = None
        self.vectorstore = None
        self.site_scribber = None
        self.list_of_links = None


    def scrape_links_from_website(self):
        """
        Scrapes links from a website, ensuring they belong to the same domain
        and applying filtering rules.

        Args:
            website_url (str): The URL of the website to scrape.

        Returns:
            list: A list of unique, filtered URLs found on the website.
                    Returns an empty list on error.
        """
        try:
            print(f"Searching [{self.website_url}] for related URLs...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
            }
            response = requests.get(self.website_url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        all_links = [a.get('href') for a in soup.find_all('a')]

        parsed_url = urlparse(self.website_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        valid_links = []
        for href in all_links:
            absolute_url = urljoin(self.website_url, href)
            if (absolute_url.startswith(base_url) and
                not absolute_url.startswith('#') and
                not absolute_url.startswith('tel:') and
                'blog' not in absolute_url and
                not absolute_url.lower().endswith(('.jpg', '.png', '.gif', '.jpeg')) and
                'mailto:' not in absolute_url):
                    valid_links.append(absolute_url)

        unique_valid_links = list(dict.fromkeys(valid_links))

        print(f"\nWebsite: {self.website_url}\nFollowing links were found within it:")
        for link in unique_valid_links:
            print(link)
        return unique_valid_links

    def load_webpages(self):
        loader = WebBaseLoader(web_paths=self.list_of_links)
        docs = loader.load()

        md = MarkdownifyTransformer(strip="a")
        converted_docs = md.transform_documents(docs)

        print("Markdown docs: ",converted_docs[2])

        print(f"Splitting page contents into smaller pieces for meaningful storage and retrieval...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        splits = text_splitter.split_documents(converted_docs)

        return splits


    def embed_and_store_docs(self):
        print(f"Extracting page contents from the website [{self.website_url}] ...")
        docs_to_store = self.load_webpages()

        print(f"Converting website contents into vectors and storing in vector DB.\nThis could take few minutes, please wait...")
        self.vectorstore = FAISS.from_documents(documents=docs_to_store, embedding=embeddings_model)
        print(f"Website content stored and ready for use!")

    
    def setup_rag_pipeline(self):

        # Retrieve and generate using the relevant snippets of the blog.
        retriever = self.vectorstore.as_retriever()

        rag_prompt_template = f"""You are an AI assistant representing the `{self.website_url}` website. When answering questions, you should directly address the user as if you are providing information about the site itself. Avoid phrases that suggest you are analyzing external context or referring to documents. Your response should feel like you are one of the support executive of the {self.website_url}.

Follow this step-by-step process to arrive at the answer:
1. First, summarize all below given context.
2. Identify key details relevant to the question from the summary of the context.
3. Based on those details, craft a precise and informative answer to the question.
4. If question cannot be answered by the given context, just say that you don't know the answer instead of assuming the answer.

Context:
{{context}}

Question:
{{question}}

Answer:"""
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

        if len(self.list_of_links) <= 0:
            print("Cannot process further as website doesn't allow scraping!")
            sys.exit(1)

        print(f"Initiating training for website: [{self.website_url}]...")
        self.embed_and_store_docs()

        print(f"Almost done, finalizing things...")
        self.setup_rag_pipeline()
        print(f"Done! Training completed! You can go ahead and ask questions about your website [{self.website_url}]")


    def ask_site_scribber(self, user_query):
        print(f"Asking Site Scribber about user query: `{user_query}`")
        print(self.site_scribber)
        response = self.site_scribber.invoke(user_query)
        return response
    

if __name__ == '__main__':
    ws = WebsiteScribber()
    ws.train_on_website('https://my-cpe.com')

    resp = ws.ask_site_scribber("Sample input query")
    print(resp)