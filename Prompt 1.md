Tu es un ingénieur spécialisé en Python, traitement documentaire et RAG juridique.

Je travaille sur un projet de moteur de recherche juridique basé sur le RAG. Ta mission est de développer la partie ingestion, préparation des documents, chunking et indexation locale.

### Objectif
Préparer les documents juridiques pour permettre à un autre module de retrouver, à partir d'une question, les circulaires pertinentes ainsi que les lois et les textes juridiques associés.

### Tâches
1. Examiner le projet existant et les documents disponibles avant de modifier le code.
2. Extraire le texte des PDF et DOCX, avec OCR si nécessaire pour les documents scannés.
3. Identifier les types de documents : circulaire, loi, décret, arrêté et autres textes.
4. Préserver les titres, les numéros d'articles, les pages, les dates et les références juridiques.
5. Implémenter trois stratégies distinctes :
   - Chunking structurel avec métadonnées.
   - RAG hiérarchique.
   - Parent-Child Chunking.
6. Générer localement des embeddings avec Python et un modèle adapté aux langues du corpus.
7. Stocker les vecteurs dans FAISS ou ChromaDB et les métadonnées dans SQLite ou une solution adaptée.
8. Conserver les relations entre les passages, les articles et les documents originaux.
9. Permettre de réindexer les documents sans créer de doublons.
10. Ajouter des tests unitaires et une documentation.

### Interface obligatoire avec l'autre membre
Fournir une interface Python permettant de :
- Charger les documents indexés.
- Rechercher les passages candidats à partir d'une requête vectorielle.
- Récupérer un passage par son identifiant.
- Récupérer son article parent et ses métadonnées.
- Accéder au document original.
- Obtenir les références juridiques extraites, avec leur provenance.

Définir un schéma commun contenant au minimum document_id, document_type, title, chunk_id, parent_id, text, source_file, page_number, article_number et references, lorsque ces informations sont disponibles.

### Évaluation
Permettre de construire un index distinct pour chacune des trois stratégies afin que l'autre membre puisse comparer leurs résultats sur les mêmes questions.

Ne pas inventer de métadonnées absentes des documents. Distinguer les informations extraites automatiquement de celles qui ont été vérifiées.

Commencer par examiner le projet, proposer l'architecture, puis implémenter progressivement et tester chaque composant. Ne pas développer l'interface utilisateur complète dans cette partie.