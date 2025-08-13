from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from .detection import analyze_frame

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

@app.post('/api/analyze')
async def analyze(frame: UploadFile = File(...), session_id: str = Form('default')):
    content = await frame.read()
    try:
        result = analyze_frame(content, session_id=session_id)
        return result
    except Exception as e:
        return {'error': str(e)}

@app.get('/health')
async def health():
    return {'status': 'ok'}
