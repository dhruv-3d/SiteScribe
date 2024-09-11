from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_pipeline import WebsiteScribber


app = FastAPI()
ws = WebsiteScribber()

@app.post("/train-on-website")
async def train_on_website(request: Request):
    data = await request.json()
    url = data.get('url')

    if url:
        ws.train_on_website()
        return JSONResponse(content={"message": "Successfully trained on the website, you can now ask questions about" + url})
    else:
        return JSONResponse(content={"error": "URL not provided."}, status_code=400)

@app.post("/get-response")
async def get_ai_response(request: Request):
    data = await request.json()
    user_query = data.get('query')

    if user_query:
        response = ws.ask_site_scribber(user_query)

        return JSONResponse(content={"response": response})
    else:
        return JSONResponse(content={"error": "Response not available for this URL."}, status_code=404)