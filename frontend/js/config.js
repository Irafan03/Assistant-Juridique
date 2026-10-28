/**
 * config.js — Configuration centralisée du frontend.
 * L'URL de base de l'API est le seul endroit à modifier pour
 * pointer vers un autre serveur (staging, production…).
 */

export const CONFIG = {
  /** URL de base de l'API FastAPI.
   *  En développement, l'API tourne sur la même origine que le frontend.
   *  Modifier uniquement cette valeur pour changer l'environnement. */
  API_BASE: '',

  /** Délai maximum (ms) avant d'abandonner une requête. */
  REQUEST_TIMEOUT_MS: 60_000,

  /** Stratégies disponibles — valeurs exactes acceptées par le backend. */
  STRATEGIES: [
    { value: 'structural',   label: 'Structurel',      description: 'Découpage par articles' },
    { value: 'hierarchical', label: 'Hiérarchique',    description: 'Arbre doc → section → article' },
    { value: 'parent_child', label: 'Parent / enfant', description: 'Larges segments + sous-segments' },
  ],

  /** Modes de recherche — valeurs exactes acceptées par le backend. */
  RETRIEVAL_MODES: [
    { value: 'hybrid',   label: 'Hybride',    description: 'Vectoriel + BM25 (RRF)' },
    { value: 'semantic', label: 'Sémantique', description: 'Vectoriel seul' },
  ],

  /** Nombre de résultats demandés par défaut. */
  DEFAULT_LIMIT: 10,
};
