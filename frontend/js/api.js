/**
 * api.js — Couche d'accès aux endpoints FastAPI réels.
 *
 * Routes documentées :
 *   POST /api/search                     → SearchResponse
 *   GET  /api/documents/{id}             → { document_id, source_file }
 *   GET  /api/documents/{id}/file        → fichier original (si configuré)
 *
 * Aucun endpoint fictif n'est appelé.
 */

import { CONFIG } from './config.js';

/** Erreur réseau ou HTTP enrichie avec le code de statut. */
export class ApiError extends Error {
  constructor(message, status = null, detail = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Wrapper fetch avec timeout et gestion d'erreur uniforme.
 * @param {string} path   — chemin relatif (ex: '/api/search')
 * @param {RequestInit} options
 * @returns {Promise<any>}
 */
async function apiFetch(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(CONFIG.API_BASE + path, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!res.ok) {
      let detail = null;
      try {
        const body = await res.json();
        detail = body.detail || null;
      } catch (_) { /* réponse non-JSON */ }
      throw new ApiError(
        detail || `Erreur HTTP ${res.status}`,
        res.status,
        detail,
      );
    }
    return res.json();
  } catch (err) {
    clearTimeout(timer);
    if (err.name === 'AbortError') {
      throw new ApiError(`Délai d'attente dépassé (${CONFIG.REQUEST_TIMEOUT_MS / 1000}s)`);
    }
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Erreur réseau : ${err.message}`);
  }
}

/**
 * Recherche juridique.
 *
 * @param {object} params
 * @param {string}  params.query           — question juridique
 * @param {string}  [params.strategy]      — 'structural' | 'hierarchical' | 'parent_child'
 * @param {string}  [params.retrieval_mode] — 'hybrid' | 'semantic'
 * @param {number}  [params.limit]
 * @param {boolean} [params.generate_answer]
 * @returns {Promise<SearchResponse>}
 */
export async function search({
  query,
  strategy   = 'structural',
  retrieval_mode = 'hybrid',
  limit      = CONFIG.DEFAULT_LIMIT,
  generate_answer = false,
}) {
  return apiFetch('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, strategy, retrieval_mode, limit, generate_answer }),
  });
}

/**
 * Métadonnées d'un document source.
 * Retourne { document_id, source_file } ou lève une ApiError.
 *
 * @param {string} documentId
 * @param {string} [strategy]
 * @returns {Promise<{ document_id: string, source_file: string }>}
 */
export async function getDocument(documentId, strategy = 'structural') {
  const qs = new URLSearchParams({ strategy }).toString();
  return apiFetch(`/api/documents/${encodeURIComponent(documentId)}?${qs}`);
}

/**
 * URL de téléchargement du fichier original.
 * Renvoie l'URL directement (pas de fetch — c'est un lien).
 * L'endpoint n'est disponible que si document_root est configuré côté serveur.
 *
 * @param {string} documentId
 * @param {string} [strategy]
 * @returns {string}
 */
export function documentFileUrl(documentId, strategy = 'structural') {
  return `${CONFIG.API_BASE}/api/documents/${encodeURIComponent(documentId)}/file?strategy=${strategy}`;
}
