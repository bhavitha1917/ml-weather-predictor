"""
main.py
Root entrypoint to run the ML Weather Predictor application.
Starts the FastAPI server (serving both API endpoints and the frontend dashboard)
and opens the browser automatically.

Usage:
    python main.py
"""

import sys
import os
import webbrowser
import threading
import time
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    url = f"http://{host}:{port}"

    # Plain ASCII only: a redirected Windows console (cp1252) cannot encode emoji/bullets
    print("=" * 65)
    print("STARTING ML WEATHER PREDICTOR & DASHBOARD")
    print("=" * 65)
    print(f"Backend API & Dashboard URL: {url}")
    print("Supported Endpoints:")
    print("  - GET /                 -> Dashboard")
    print("  - GET /health           -> Service health check")
    print("  - GET /cities           -> List of trained cities")
    print("  - GET /predict?city=... -> Live prediction with 80% interval")
    print("  - GET /metrics?city=... -> Test-year metrics and interval coverage")
    print("  - GET /simulate?...     -> Illustrative what-if tool")
    print("  - GET /figures/...      -> EDA and residual plots")
    print("  - GET /docs             -> Interactive API documentation")
    print("=" * 65)

    def open_browser():
        time.sleep(1.5)
        print(f"Opening browser at {url}...")
        webbrowser.open(url)

    # Launch browser in a background thread if running locally
    if os.environ.get("RENDER") is None:
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("backend.app:app", host=host, port=port, reload=False)
