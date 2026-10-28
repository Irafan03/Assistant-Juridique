/**
 * store.js — État de l'application (historique de conversations en mémoire).
 *
 * L'historique est persisté dans localStorage uniquement pour les requêtes
 * (pas les réponses complètes, qui peuvent être volumineuses).
 */

const STORAGE_KEY = 'rag_chat_history';
const MAX_HISTORY = 20;

/** @typedef {{ id: string, query: string, ts: number }} HistoryItem */

/**
 * Retourne l'historique des questions (plus récent en premier).
 * @returns {HistoryItem[]}
 */
export function getHistory() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch (_) {
    return [];
  }
}

/**
 * Ajoute une question à l'historique.
 * @param {string} query
 * @returns {string} id généré
 */
export function addToHistory(query) {
  const history = getHistory();
  const id = `h${Date.now()}`;
  history.unshift({ id, query: query.trim(), ts: Date.now() });
  const trimmed = history.slice(0, MAX_HISTORY);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch (_) { /* quota dépassé */ }
  return id;
}

/**
 * Supprime une entrée de l'historique.
 * @param {string} id
 */
export function removeFromHistory(id) {
  const history = getHistory().filter(item => item.id !== id);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
  } catch (_) {}
}

/**
 * Formate une date relative (ex: "il y a 2 min").
 * @param {number} ts
 * @returns {string}
 */
export function relativeTime(ts) {
  const diff = Date.now() - ts;
  const min  = Math.floor(diff / 60_000);
  const h    = Math.floor(diff / 3_600_000);
  const d    = Math.floor(diff / 86_400_000);
  if (min  < 1)  return 'à l\'instant';
  if (min  < 60) return `il y a ${min} min`;
  if (h    < 24) return `il y a ${h}h`;
  return `il y a ${d}j`;
}
