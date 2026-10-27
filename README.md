# Assistant juridique — retrieval RAG

Cette partie implémente la recherche et la présentation des résultats demandées dans `Prompt 2.md`. Elle consomme les index construits par le module d’ingestion de Prompt 1 et ne crée ni index vectoriel ni embeddings.

## État du workspace

Le dépôt fourni ne contient actuellement que les deux prompts et un rapport PDF sur FlowForge, un projet d’automatisation sans rapport avec le corpus juridique. Il n’y a ni corpus juridique, ni indexation, ni interface du premier membre. Le code ci-dessous définit donc le contrat d’adaptation. Aucune relation juridique ni mesure expérimentale n’est préremplie. Les 20 questions et leurs réponses attendues doivent être annotées à partir des documents réels avant de lancer l’évaluation.

## Contrat avec le module d’indexation

Fournir un objet conforme à `legal_rag.interfaces.IndexAdapter` :

- `search(query, strategy, limit)` : recherche sémantique dans l’index de la stratégie indiquée.
- `iter_chunks(strategy)` : passages de cette stratégie, pour BM25 et les références exactes.
- `get_chunk(chunk_id, strategy)` et `get_parent(chunk, strategy)` : récupération des parents.

`strategy` vaut `structural`, `hierarchical` ou `parent_child`. Chaque élément est un `Chunk` ou un dictionnaire converti par `Chunk.from_mapping`. Le schéma conserve les champs communs de Prompt 1, ainsi que `verified_metadata`, qui identifie les valeurs confirmées humainement. Les champs inconnus restent absents : le retrieval ne complète pas les métadonnées manquantes.

Exemple minimal d’adaptation :

```python
from legal_rag.api import create_app
from legal_rag.models import Chunk
from legal_rag.retrieval import LegalRetriever

class Prompt1Adapter:
    def __init__(self, prompt1_index):
        self.index = prompt1_index

    def search(self, query, *, strategy, limit):
        return [Chunk.from_mapping(row) for row in self.index.search(query, strategy=strategy, k=limit)]

    def iter_chunks(self, *, strategy):
        return [Chunk.from_mapping(row) for row in self.index.iter_chunks(strategy=strategy)]

    def get_chunk(self, chunk_id, *, strategy):
        row = self.index.get_chunk(chunk_id, strategy=strategy)
        return Chunk.from_mapping(row) if row else None

    def get_parent(self, chunk, *, strategy):
        row = self.index.get_parent(chunk.chunk_id, strategy=strategy)
        return Chunk.from_mapping(row) if row else None

adapter = Prompt1Adapter(your_index)
app = create_app(LegalRetriever(adapter), document_root="/path/to/corpus")
```

Adapter les noms d’arguments à l’API réelle du premier module. Le serveur ne peut pas ouvrir les originaux tant que `document_root` n’est pas configuré; les chemins d’index sont alors confinés à ce répertoire.

## Fonctionnement

- Recherche vectorielle fournie par l’adaptateur et BM25 local, combinés par Reciprocal Rank Fusion (RRF).
- Mode `semantic` pour la référence vectorielle seule et mode `hybrid` par défaut.
- Détection des références au format légal français, puis résolution exacte contre les références effectivement présentes dans les chunks et leurs métadonnées.
- Les relations explicites entre circulaires et textes sont étiquetées `explicit_reference`; les voisins issus du classement restent `semantic_similarity`.
- Un reranker peut être fourni via `Reranker`. Aucun modèle externe ou appel LLM n’est imposé.
- Le générateur optionnel reçoit uniquement les passages retenus comme preuves. Les citations et leur rendu final doivent pointer vers les `citation_id` et les passages transmis. Sans générateur configuré, l’API renvoie les extraits et ne fabrique pas de réponse juridique.
- L’interface web est servie sur `/`; API de recherche sur `POST /api/search`, originaux sur `GET /api/documents/{document_id}/file`.

## Lancement

```powershell
python -m pip install -e ".[dev]"
uvicorn your_adapter_module:app --reload
```

Le module montré dans l’exemple doit exporter `app` après avoir construit l’adaptateur. Le `app` fourni par `legal_rag.api` sans adaptateur répond avec HTTP 503, afin de signaler l’absence d’index configuré.

Requête API :

```json
{"query":"Quelles règles la circulaire applique-t-elle au congé ?","strategy":"structural","retrieval_mode":"hybrid","limit":10,"generate_answer":false}
```

## Évaluation sur le corpus

Créer `evaluation/cases.json` avec exactement 20 questions. Chaque liste de pertinence doit contenir des identifiants `chunk_id` vérifiés dans les index :

```json
[
  {
    "id": "q01",
    "query": "Question vérifiée sur un document réel",
    "relevant_circulars": ["chunk-circulaire-1"],
    "relevant_laws": ["chunk-loi-4"],
    "relevant_articles": ["chunk-article-17"]
  }
]
```

Les métriques sont calculées séparément pour circulaires, textes associés et articles : Precision@k, Recall@k, MRR et nDCG@k. `run_evaluation` accepte des configurations nommées, par exemple une entrée par stratégie/mode et une variante avec reranker activé. Comparer les résultats seulement après annotation des jugements attendus depuis le corpus. Le script ne fournit volontairement aucun score par défaut.

## Vérification

```powershell
python -m pytest
```

Les tests couvrent les métriques, les références explicites, les parents et le lien vers un texte associé. Ils n’évaluent pas la qualité d’un index juridique absent de ce workspace.
