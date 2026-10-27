from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from pathlib import Path
from pydantic import BaseModel, Field

from .interfaces import STRATEGIES
from .retrieval import LegalRetriever


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    strategy: str = "structural"
    retrieval_mode: str = "hybrid"
    limit: int = Field(default=10, ge=1, le=50)
    generate_answer: bool = False


def create_app(retriever: LegalRetriever | None = None, *, document_root: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Assistant juridique — Retrieval", version="0.1.0")
    app.state.retriever = retriever
    app.state.document_root = Path(document_root).resolve() if document_root else None

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return """<!doctype html><html lang='fr'><meta charset='utf-8'><title>Recherche juridique</title>
<style>body{font:16px system-ui;max-width:900px;margin:3rem auto;padding:0 1rem;color:#202124}textarea{width:100%;height:90px}button,select{padding:.6rem;margin:.5rem 0}article{border-top:1px solid #ddd;padding:1rem 0}small{color:#555}</style>
<h1>Recherche juridique</h1><p>Interrogez les passages indexés. Les liens et références dépendent des métadonnées du corpus.</p>
<textarea id='q' placeholder='Votre question en français ou en arabe'></textarea><br><select id='strategy'><option value='structural'>Structurel</option><option value='hierarchical'>Hiérarchique</option><option value='parent_child'>Parent / enfant</option></select><select id='mode'><option value='hybrid'>Recherche hybride</option><option value='semantic'>Vectorielle seule</option></select><button onclick='run()'>Rechercher</button><div id='out'></div>
<script>async function run(){const out=document.querySelector('#out');out.textContent='Recherche…';try{const r=await fetch('/api/search',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({query:document.querySelector('#q').value,strategy:document.querySelector('#strategy').value,retrieval_mode:document.querySelector('#mode').value,generate_answer:true})});const d=await r.json();if(!r.ok)throw Error(d.detail||'Erreur');out.innerHTML='';if(d.answer){const answer=document.createElement('article'),h=document.createElement('h2'),p=document.createElement('p');h.textContent='Réponse';p.textContent=d.answer;answer.append(h,p);out.append(answer)}for(const [title,items] of [['Circulaires',d.circulaires],['Textes associés',d.textes_associes],['Articles',d.articles],['Passages',d.passages]]){const h=document.createElement('h2');h.textContent=title;out.append(h);for(const x of items){const a=document.createElement('article'),t=document.createElement('p'),m=document.createElement('small');t.textContent=x.text;m.textContent=[x.citation_id,x.title,x.article_number,x.source_file?'Fichier original':null,x.page_number?'p. '+x.page_number:null,x.relation].filter(Boolean).join(' · ');a.append(m,t);if(x.source_file){const link=document.createElement('a');link.textContent='Ouvrir le document';link.href='/api/documents/'+encodeURIComponent(x.document_id)+'/file';a.append(link)}out.append(a)}}for(const w of d.warnings||[]){const p=document.createElement('p');p.textContent=w;out.prepend(p)}}catch(e){out.textContent=e.message}}</script></html>"""

    @app.post("/api/search")
    def search(request: SearchRequest) -> dict[str, Any]:
        engine: LegalRetriever | None = app.state.retriever
        if engine is None:
            raise HTTPException(503, "Index non configuré. Fournissez un IndexAdapter via create_app().")
        if request.strategy not in STRATEGIES:
            raise HTTPException(422, f"strategy doit être l’une de : {', '.join(STRATEGIES)}")
        try:
            return engine.search(request.query, strategy=request.strategy, retrieval_mode=request.retrieval_mode, limit=request.limit, generate_answer=request.generate_answer).to_dict()
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/documents/{document_id}")
    def original_document(document_id: str, strategy: str = Query(default="structural")) -> dict[str, str]:
        engine: LegalRetriever | None = app.state.retriever
        if engine is None:
            raise HTTPException(503, "Index non configuré.")
        if strategy not in STRATEGIES:
            raise HTTPException(422, "Stratégie inconnue.")
        matches = [c for c in engine.index.iter_chunks(strategy=strategy) if c.document_id == document_id and c.source_file]
        if not matches:
            raise HTTPException(404, "Document original introuvable dans les métadonnées indexées.")
        # Return the stored path; clients must use a deployment-specific, access-
        # controlled document server to serve files. Never expose arbitrary paths.
        return {"document_id": document_id, "source_file": matches[0].source_file or ""}

    @app.get("/api/documents/{document_id}/file")
    def open_document(document_id: str, strategy: str = Query(default="structural")) -> FileResponse:
        engine: LegalRetriever | None = app.state.retriever
        root: Path | None = app.state.document_root
        if engine is None or root is None:
            raise HTTPException(404, "Le serveur de fichiers originaux n’est pas configuré.")
        if strategy not in STRATEGIES:
            raise HTTPException(422, "Stratégie inconnue.")
        matches = [c for c in engine.index.iter_chunks(strategy=strategy) if c.document_id == document_id and c.source_file]
        if not matches:
            raise HTTPException(404, "Document original introuvable.")
        candidate = Path(matches[0].source_file or "")
        candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise HTTPException(404, "Fichier absent du répertoire documentaire autorisé.")
        return FileResponse(candidate)

    return app


app = create_app()
