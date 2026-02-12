"""
Simple backend starter that delays model loading
"""
import uvicorn

if __name__ == "__main__":
    print("Starting SEFS Backend...")
    print("Note: First startup will download embedding model (~80MB)")
    print("This may take 1-2 minutes...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
