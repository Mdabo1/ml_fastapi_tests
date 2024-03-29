from fastapi import FastAPI, HTTPException, Query, Depends
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    """
    Represents an input item for sentiment analysis.
    
    Attributes:
        text (str): The text to be analyzed for sentiment.
    """
    text: str

app = FastAPI()  # Initialize FastAPI app

def get_classifier():
    """
    Initialize sentiment-analysis pipeline.
    """
    return pipeline("sentiment-analysis")

@app.get("/")  # Adding GET method
def root():
    """
    Root endpoint returning a simple message.
    """
    return {"message": "Hello World"}

@app.post("/predict/")  # Adding POST method
def predict(item: Item, classifier=Depends(get_classifier)):
    """
    Endpoint to predict sentiment analysis for the given text.
    
    Args:
        item (Item): Input item containing text to analyze.
        classifier (transformers.Pipeline): Sentiment analysis pipeline.
        
    Returns:
        dict: Result of sentiment analysis.
    """
    if not item.text.strip():  # Check if text is empty or contains only whitespace
        raise HTTPException(status_code=400, detail="Text input cannot be empty or whitespace.")
    try:
        result = classifier(item.text)[0]
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
