import sys
import uvicorn
from config import SERVER_HOST, SERVER_PORT

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    print(f"Starting HCMC AI Video Retrieval Server on http://{SERVER_HOST}:{SERVER_PORT} ...")
    uvicorn.run("src.api.main:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
