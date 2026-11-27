import os, json, uuid, shutil, subprocess, tempfile, math, re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from openai import OpenAI

load_dotenv()
API_KEY=os.getenv("OPENAI_API_KEY")
MODEL=os.getenv("OPENAI_CHAT_MODEL","gpt-4.1-mini")
MAX_MB=int(os.getenv("MAX_UPLOAD_MB","500"))
BASE=Path(__file__).parent
WORK=BASE/"jobs"; WORK.mkdir(exist_ok=True)
app=FastAPI(title="ViralCut AI")
client=OpenAI(api_key=API_KEY) if API_KEY else None

def run(cmd):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    if p.returncode: raise RuntimeError(p.stderr[-2500:])
    return p.stdout.strip()

def duration(path):
    return float(run(["ffprobe","-v","error","-show_entries","format=duration","-of","default=noprint_wrappers=1:nokey=1",str(path)]))

def safe_name(s):
    return re.sub(r"[^a-zA-Z0-9_-]+","_",s)[:50] or "video"

def extract_audio_chunks(video, folder, chunk_seconds=600):
    dur=duration(video); chunks=[]
    for start in range(0,int(math.ceil(dur)),chunk_seconds):
        out=folder/f"audio_{start}.mp3"
        run(["ffmpeg","-y","-ss",str(start),"-i",str(video),"-t",str(chunk_seconds),"-vn","-ac","1","-ar","16000","-b:a","48k",str(out)])
        if out.exists() and out.stat().st_size:
            chunks.append((start,out))
    return chunks,dur

def transcribe(video, folder):
    chunks, dur=extract_audio_chunks(video,folder)
    texts=[]
    for offset,audio in chunks:
        if audio.stat().st_size>24*1024*1024:
            raise RuntimeError("Áudio de um bloco excedeu 24 MB. Tente um vídeo menor.")
        with audio.open("rb") as f:
            r=client.audio.transcriptions.create(model="gpt-transcribe",file=f)
        texts.append({"offset":offset,"text":r.text})
    return texts,dur

def pick_moments(transcripts, video_duration, clip_len, quantity, mode):
    transcript="\n\n".join(f"[INÍCIO DO BLOCO {t['offset']}s]\n{t['text']}" for t in transcripts)
    # limit transcript context; clips are still selected based on the provided text
    transcript=transcript[:90000]
    prompt=f"""Você é editor especialista em Shorts. Analise esta transcrição e selecione até {quantity} trechos com potencial de retenção/viralização, modo {mode}.
Cada trecho deve durar aproximadamente {clip_len} segundos, estar dentro do vídeo de {video_duration:.1f}s e começar/end em limites naturais de fala.
Priorize gancho nos primeiros segundos, clareza sem contexto externo, emoção, humor, opinião forte, informação útil ou história com payoff. Não invente falas.
Retorne APENAS JSON: {{ "clips":[{{"start":0,"end":{clip_len},"title":"título curto","hook":"gancho","reason":"motivo","score":90}}] }}.
A duração real de cada corte deve ser {clip_len} segundos (tolerância 3s), ajuste start para respeitar a duração do vídeo.
Transcrição com offsets dos blocos:
{transcript}"""
    response=client.chat.completions.create(model=MODEL,temperature=0.25,response_format={"type":"json_object"},messages=[{"role":"system","content":"Você seleciona trechos de vídeos. Retorne JSON válido e nunca invente timestamps além do conteúdo contextualizado."},{"role":"user","content":prompt}])
    data=json.loads(response.choices[0].message.content)
    clips=data.get("clips",[])
    clean=[]
    for c in clips:
        try:
            start=max(0,float(c["start"])); end=min(video_duration,float(c["end"]))
            if end-start < clip_len-3: end=min(video_duration,start+clip_len)
            if end-start > clip_len+3: end=start+clip_len
            start=max(0,min(start,video_duration-clip_len)); end=min(video_duration,start+clip_len)
            if end-start < min(clip_len*.65,12): continue
            clean.append({"start":start,"end":end,"title":str(c.get("title","Melhor momento"))[:100],"hook":str(c.get("hook",""))[:180],"reason":str(c.get("reason",""))[:250],"score":max(0,min(100,int(c.get("score",70))))})
        except: continue
    # avoid overlapping duplicates
    clean.sort(key=lambda x:x["score"],reverse=True)
    unique=[]
    for c in clean:
        if all(abs(c["start"]-u["start"])>clip_len*.55 for u in unique): unique.append(c)
    return sorted(unique[:quantity],key=lambda x:x["start"])

def make_ass(text, length, out):
    # approximate caption timing: split text into readable chunks, distribute over clip duration
    words=re.findall(r"\S+",text)
    if not words: words=[""]
    chunks=[]
    for i in range(0,len(words),5): chunks.append(" ".join(words[i:i+5]))
    per=max(0.8,length/max(1,len(chunks)))
    def ts(t):
        h=int(t//3600); m=int((t%3600)//60); s=t%60
        return f"{h}:{m:02d}:{s:05.2f}"
    lines=["[Script Info]","ScriptType: v4.00+","PlayResX: 1080","PlayResY: 1920","WrapStyle: 2","",
    "[V4+ Styles]","Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    "Style: Default,Arial,64,&H00FFFFFF,&H0000FFFF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,1,5,2,2,70,70,240,1","",
    "[Events]","Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for i,ch in enumerate(chunks):
        start=i*per; end=min(length,(i+1)*per)
        escaped=ch.replace("{","\\{").replace("}","\\}").replace("\\n"," ")
        lines.append(f"Dialogue: 0,{ts(start)},{ts(max(start+.25,end))},Default,,0,0,0,,{escaped}")
    out.write_text("\n".join(lines),encoding="utf-8")

def download_youtube(url, folder):
    # Only accept public YouTube URLs; use yt-dlp to fetch media the user is authorized to edit.
    from urllib.parse import urlparse
    host=(urlparse(url).hostname or "").lower()
    allowed={"youtube.com","www.youtube.com","m.youtube.com","youtu.be","www.youtu.be","youtube-nocookie.com","www.youtube-nocookie.com"}
    if host not in allowed:
        raise RuntimeError("Link inválido. Use um endereço youtube.com ou youtu.be.")
    out=folder/"youtube_source.%(ext)s"
    cmd=["python","-m","yt_dlp","--no-playlist","--format","bestvideo[height<=1080]+bestaudio/best[height<=1080]","--merge-output-format","mp4","--output",str(out),url]
    try:
        run(cmd)
        candidates=list(folder.glob("youtube_source.*"))
        candidates=[p for p in candidates if p.suffix.lower() in {".mp4",".mkv",".webm",".mov"}]
        if not candidates: raise RuntimeError("O download terminou, mas não encontrei o vídeo baixado.")
        return candidates[0]
    except RuntimeError as e:
        msg=str(e)
        if "yt_dlp" in msg or "No module named" in msg:
            raise RuntimeError("O downloader não está instalado. No terminal, rode: python -m pip install -r requirements.txt")
        raise RuntimeError("Não foi possível baixar esse vídeo do YouTube. Verifique se o link é público, se o vídeo permite acesso e tente novamente. Detalhe: "+msg[-1000:])

def render_clip(video, clip, index, folder, transcript):
    start=clip["start"]; length=clip["end"]-start
    out=folder/f"cut_{index:02d}.mp4"; ass=folder/f"sub_{index:02d}.ass"
    # use the nearest chunk's text as caption source; a production version should align transcript timestamps
    closest=min(transcript,key=lambda t:abs(t["offset"]-start))
    make_ass(closest["text"],length,ass)
    vf=f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,subtitles={str(ass).replace(chr(92),'/').replace(':','\\\\:')}"
    run(["ffmpeg","-y","-ss",str(start),"-i",str(video),"-t",str(length),"-vf",vf,"-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k","-movflags","+faststart",str(out)])
    return out

@app.get("/",response_class=HTMLResponse)
def home():
    return (BASE/"index.html").read_text(encoding="utf-8")

@app.post("/api/process")
async def process(video: Optional[UploadFile]=File(None), youtube_url:str=Form(""), seconds:int=Form(30), quantity:str=Form("10"), mode:str=Form("Mais viral"), caption_style:str=Form("Viral")):
    if not client: raise HTTPException(500,"Configure OPENAI_API_KEY no arquivo .env.")
    if seconds not in (30,60): raise HTTPException(400,"Duração deve ser 30 ou 60.")
    try:
        match=re.search(r"\d+", quantity)
        n=int(match.group(0)) if match else 10
        n=max(1,min(n,100))
    except Exception:
        n=10
    if not video or not video.filename:
        if not youtube_url.strip(): raise HTTPException(400,"Envie um arquivo de vídeo ou cole um link do YouTube.")
    job=str(uuid.uuid4()); folder=WORK/job; folder.mkdir()
    try:
        if video and video.filename:
            ext=Path(video.filename).suffix or ".mp4"; source=folder/("source"+ext)
            data=await video.read()
            if len(data)>MAX_MB*1024*1024: raise HTTPException(413,f"Arquivo acima de {MAX_MB}MB.")
            source.write_bytes(data)
        else:
            source=download_youtube(youtube_url.strip(),folder)
        if source.stat().st_size>MAX_MB*1024*1024:
            raise RuntimeError(f"Vídeo acima do limite de {MAX_MB}MB. Escolha um vídeo menor.")
        transcripts,vd=transcribe(source,folder)
        clips=pick_moments(transcripts,vd,seconds,n,mode)
        if not clips: raise RuntimeError("A IA não encontrou cortes válidos. Tente outro vídeo ou modo.")
        results=[]
        for i,c in enumerate(clips,1):
            path=render_clip(source,c,i,folder,transcripts)
            c["file"]=f"/download/{job}/{path.name}"
            c["format"]="MP4"
            results.append(c)
        return {"job":job,"video_duration":vd,"clips":results,"message":f"{len(results)} cortes MP4 gerados."}
    except Exception as e:
        shutil.rmtree(folder,ignore_errors=True)
        raise HTTPException(500,str(e))

@app.get("/download/{job}/{filename}")
def download(job,filename):
    p=WORK/job/filename
    if not p.exists() or p.suffix.lower()!=".mp4": raise HTTPException(404,"Arquivo não encontrado")
    return FileResponse(p,media_type="video/mp4",filename=filename)

@app.get("/health")
def health():
    return {"ok":True,"api_key_configured":bool(API_KEY)}
