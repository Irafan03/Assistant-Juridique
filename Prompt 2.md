Tu es un ingénieur spécialisé en recherche d'information, RAG et systèmes juridiques.

Je développe un moteur de recherche juridique basé sur le RAG avec un autre membre du groupe. Celui-ci développe l'ingestion documentaire, les trois stratégies de chunking et la base vectorielle locale.

Ta mission est de développer la partie retrieval, identification des relations juridiques, API, affichage des résultats et évaluation.

### Objectif fonctionnel
L'utilisateur pose une question juridique en français ou en arabe, selon le corpus. Le système doit afficher :
1. Les circulaires pertinentes pour la question.
2. Les lois, décrets, arrêtés et autres textes associés.
3. Les articles et passages pertinents.
4. Les extraits exacts et les références des documents.
5. Les liens permettant d'ouvrir les fichiers originaux.

Le RAG doit aussi pouvoir générer une réponse synthétique fondée sur les passages récupérés, accompagnée de citations vérifiables, sans inventer d'informations juridiques.

### Tâches
1. Définir une interface claire avec le module d'indexation de l'autre membre. Ne pas recréer sa base vectorielle ni dupliquer son code.
2. Implémenter une recherche sémantique sur les index fournis.
3. Ajouter une recherche lexicale BM25, une recherche exacte des références juridiques et une fusion des classements.
4. Ajouter un reranking des meilleurs candidats si cela améliore les résultats.
5. Récupérer les passages parents et leur contexte hiérarchique.
6. Identifier les circulaires correspondant à la question.
7. Retrouver les lois associées à partir des références explicites, des métadonnées fiables et des relations vérifiables.
8. Distinguer les références explicites des simples similarités sémantiques.
9. Construire le contexte transmis au LLM et produire une réponse fondée sur les sources récupérées.
10. Afficher séparément les circulaires, les lois associées, les articles, les extraits et les fichiers originaux.
11. Créer une API avec FastAPI et une interface utilisateur simple.
12. Ajouter des tests, un README et une documentation.

### Évaluation sur 20 questions
Construire un jeu de test dont les documents et articles attendus sont vérifiés à partir du corpus réel.

Comparer :
- Les trois stratégies de chunking fournies par l'autre membre.
- La recherche vectorielle seule et la recherche hybride.
- Le reranking avec et sans reranking.

Calculer Precision@k, Recall@k, MRR et nDCG@k. Évaluer séparément la récupération des circulaires, des lois associées et des articles pertinents. Ne pas inventer les résultats expérimentaux.

### Contraintes
- Respecter l'architecture et le schéma de données communs.
- Ne pas modifier arbitrairement le module d'indexation de l'autre membre.
- Ne jamais inventer de références juridiques.
- Signaler les résultats insuffisamment documentés.
- Garder une trace des sources utilisées pour chaque résultat et chaque réponse générée.

Commencer par examiner le projet existant et définir le contrat d'interface avec l'autre membre avant d'implémenter le moteur. Tester les composants séparément, puis préparer leur intégration.