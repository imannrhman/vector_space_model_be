from fastapi import FastAPI, status
from search import *
from fastapi.middleware.cors import CORSMiddleware
from data import korpus_data
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/search")
async def main(query:str):
    try :
        if (query) :
            results = search_vector_model(korpus_data, query)
            if results :
                return JSONResponse(results, status_code= status.HTTP_200_OK)
            else :
                
                return JSONResponse({"messages" : "Query tidak ditemukan pada korpus"}, status_code=status.HTTP_404_NOT_FOUND)
        else :
            return JSONResponse([ {"document_id" : id, "corpus" : corpus, "scores" : 0.0} for id, corpus in korpus_data.items() ], status_code= status.HTTP_200_OK)
    except :
        return JSONResponse({"messages" : "Terjadi kesalahan pada server"}, status_code= status.HTTP_500_INTERNAL_SERVER_ERROR)
