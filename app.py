#!/usr/bin/env python
# coding: utf-8
"""
Robust NER API – v9 (CORREGIDA)
────────────────────────────────────────────────────────────────────────
✓ CORS funcional para frontend
✓ Soporte OPTIONS /ner para evitar errores preflight
✓ Token Bearer obligatorio
"""

from __future__ import annotations
import os, re, pathlib
from typing import List, Set, Dict, Any, Optional
import torch
from fastapi import FastAPI, HTTPException, Header, Depends, status
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ────────── Config ──────────
MODEL_DIR      = os.getenv("MODEL_DIR", "models")
DEVICE         = 0 if torch.cuda.is_available() else -1
STRIDE_TOKENS  = int(os.getenv("STRIDE_TOKENS", "200"))
MIN_SCORE      = float(os.getenv("MIN_SCORE", "0.5"))
MIN_LENGTH     = int(os.getenv("MIN_LENGTH", "3"))
GAP_CHARS      = int(os.getenv("GAP_CHARS", "4"))
BLACKLIST_PATH = pathlib.Path(os.getenv("BLACKLIST_PATH", "blacklist.txt"))
ADMIN_TOKEN    = os.getenv("ADMIN_TOKEN", "changeme")

SEP_TOKENS   = {",", ";", "–", "-", "—"}
AND_PATTERNS = {" and ", " & "}

DEMONYM_SUFFIXES = ("ian","ean","ite","ish","ic","tian","ese","i","an","chi","ard")
_DEM_RE  = re.compile(rf"^[A-Z][a-z]+({'|'.join(DEMONYM_SUFFIXES)})$")
def _is_demonym(txt:str)->bool:
    if _DEM_RE.match(txt): return True
    if "-" in txt:
        a,_,b = txt.partition("-")
        return _DEM_RE.match(a) or _DEM_RE.match(b)
    return False

_CLEAN_RE = re.compile(r"""^[\s([\{«"'‐-‒–—-]+|[\s)\]}»"'‐-‒–—.]+$""")
def _clean_word(t:str)->str:
    return _CLEAN_RE.sub("", t).strip()

_MISC_STOPWORDS: Set[str] = {
    "album","producer","year","nobel","laure","laureate",
    "ostinato","parallel"
}
_WHITELIST_SHORT = {"lake","wars"}

def _is_noise_misc(span:str)->bool:
    tokens = re.split(r"[ \-]", span.lower())
    if any(tok in _MISC_STOPWORDS for tok in tokens):
        return True
    last = tokens[-1]
    return len(last)<=5 and last not in _WHITELIST_SHORT

# ────────── Esquemas ──────────
class NERRequest(BaseModel):
    text:str
class Entity(BaseModel):
    entity_group:str; score:float; word:str; start:int; end:int
class NERResponse(BaseModel):
    entities:List[Entity]
class Patch(BaseModel):
    add:Optional[List[str]]=None; remove:Optional[List[str]]=None

# ────────── App FastAPI ──────────
app = FastAPI(title="Robust NER API (v9)")

# CORS – ABIERTO PARA DESARROLLO
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Autenticación por token
def _auth(tok:str=Header(None,alias="Authorization")):
    if tok!=f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED)

# ────────── Modelo ──────────
tokenizer=model=nlp=None
def _load_pipe():
    global tokenizer,model,nlp
    tokenizer=AutoTokenizer.from_pretrained(MODEL_DIR,local_files_only=True)
    model    =AutoModelForTokenClassification.from_pretrained(MODEL_DIR,local_files_only=True)
    nlp      =pipeline("ner",model=model,tokenizer=tokenizer,
                       aggregation_strategy="simple",device=DEVICE,
                       stride=STRIDE_TOKENS)

# ────────── Blacklist ──────────
BLACKLIST:Set[str]=set()
def _load_blacklist():
    if BLACKLIST_PATH.exists():
        BLACKLIST.update(l.strip().lower() for l in BLACKLIST_PATH.read_text("utf8").splitlines() if l.strip())
def _save_blacklist():
    BLACKLIST_PATH.parent.mkdir(parents=True,exist_ok=True)
    BLACKLIST_PATH.write_text("\n".join(sorted(BLACKLIST)),"utf8")

@app.on_event("startup")
def _startup():
    _load_pipe()
    _load_blacklist()

# ────────── Utilidades ──────────
def _has_blocking_sep(frag:str)->bool:
    lower=frag.lower()
    return any(sep in frag for sep in SEP_TOKENS) or any(p in lower for p in AND_PATTERNS)

def _split_composite(ent:Dict[str,Any],txt:str)->List[Dict[str,Any]]:
    work=ent["word"].replace(" and ",",")
    if "," not in work: return [ent]
    parts=[p.strip() for p in work.split(",") if p.strip()]
    out,pos=[],ent["start"]
    for p in parts:
        idx=txt.find(p,pos); pos=idx+len(p) if idx!=-1 else pos
        if idx==-1: continue
        clean=_clean_word(p)
        if clean: out.append({**ent,"word":clean,"start":idx,"end":idx+len(clean)})
    return out or [ent]

def _force_merge(prev,nxt,txt):
    if prev["entity_group"]==nxt["entity_group"]=="ORG":
        frag=txt[prev["end"]:nxt["start"]].lower()
        return ", davis" in frag and prev["word"].endswith("University of California")
    return False

def post_process(raw:List[Dict[str,Any]],txt:str)->List[Dict[str,Any]]:
    ents=[e for e in raw if e["score"]>=MIN_SCORE and len(e["word"])>=MIN_LENGTH]
    ents.sort(key=lambda e:e["start"])

    merged=[]
    for e in ents:
        if merged and ((e["entity_group"]==merged[-1]["entity_group"]
                        and e["start"]-merged[-1]["end"]<=GAP_CHARS
                        and not _has_blocking_sep(txt[merged[-1]["end"]:e["start"]]))
                       or _force_merge(merged[-1],e,txt)):
            merged[-1]["end"]=e["end"]
            merged[-1]["score"]=max(merged[-1]["score"],e["score"])
        else:
            merged.append({**e})

    final=[]
    for m in merged:
        m["word"]=_clean_word(txt[m["start"]:m["end"]])
        if _is_demonym(m["word"]): continue
        if m["entity_group"]=="MISC" and _is_noise_misc(m["word"]): continue
        if m["word"].lower() in BLACKLIST: continue
        final.extend(_split_composite(m,txt))
    return list({(e["start"],e["end"]):e for e in final}.values())

# ────────── ENDPOINTS ──────────
@app.post("/ner", response_model=NERResponse)
def ner(req:NERRequest):
    if not req.text.strip():
        raise HTTPException(400,"El campo 'text' no puede estar vacío")
    clean=post_process(nlp(req.text),req.text)
    return NERResponse(entities=[Entity(**e) for e in clean])

@app.options("/ner")  # Soluciona error OPTIONS preflight
def ner_options():
    return JSONResponse(status_code=200)

@app.get("/admin/blacklist", dependencies=[Depends(_auth)], response_model=List[str])
def bl_get(): return sorted(BLACKLIST)

@app.post("/admin/blacklist", dependencies=[Depends(_auth)], response_model=List[str])
def bl_patch(p:Patch):
    if p.add: BLACKLIST.update(t.lower() for t in p.add)
    if p.remove: BLACKLIST.difference_update(t.lower() for t in p.remove)
    _save_blacklist(); return sorted(BLACKLIST)

@app.get("/admin/misc_stopwords", dependencies=[Depends(_auth)], response_model=List[str])
def sw_get(): return sorted(_MISC_STOPWORDS)

@app.post("/admin/misc_stopwords", dependencies=[Depends(_auth)], response_model=List[str])
def sw_patch(p:Patch):
    if p.add: _MISC_STOPWORDS.update(t.lower() for t in p.add)
    if p.remove: _MISC_STOPWORDS.difference_update(t.lower() for t in p.remove)
    return sorted(_MISC_STOPWORDS)

# ────────── Run Local ──────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7999)
