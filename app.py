from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_pipeline import WebsiteScribber


app = FastAPI()
ws = WebsiteScribber()


class WebsiteToTrain(BaseModel):
    base_url: str

class UserQuery(BaseModel):
    query: str    

@app.post("/train-on-website")
async def train_on_website(website_to_train: WebsiteToTrain):

    url = website_to_train.base_url

    if url:
        ws.train_on_website(url)
        return JSONResponse(content={"message": "Successfully trained on the website, you can now ask questions about" + url})
    else:
        return JSONResponse(content={"error": "URL not provided."}, status_code=400)

@app.post("/get-response")
async def get_ai_response(user_query: UserQuery):

    user_query = user_query.query

    if user_query:
        response = ws.ask_site_scribber(user_query)

        return JSONResponse(content={"response": response})
    else:
        return JSONResponse(content={"error": "Response not available for this URL."}, status_code=404)