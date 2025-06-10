# TEMP_run_fastapi.py (Create this new temporary file)
import uvicorn
import os

# Ensure your FastAPI app is imported correctly
from fastapi_app import app # This imports the 'app' instance from fastapi_app.py

print(f"Attempting to run app from: {app.__module__}")
print(f"App type: {type(app)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)