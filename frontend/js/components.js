/**
 * components.js — Fonctions de rendu HTML pour les composants UI.
 *
 * Toutes les fonctions retournent des éléments DOM ou des fragments.
 * Aucune donnée n'est inventée : si un champ est null/undefined, il est omis.
 */

/** Échappe les caractères HTML pour éviter XSS. */
function esc(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Badge de type documentaire.
 * @param {string|null} docType
 * @returns {string} HTML
 */
export function docTypeBadge(docType) {
  if (!docType) return '';
  const map = {
    'loi': 'Loi',
    'circulaire': 'Circulaire',
    'décret': 'Décret',
    'arrêté': 'Arrêté',
    'directive': 'Directive',
    'recommandation': 'Recommandation',
    'texte': 'Texte',
  };
  const label = map[docType.toLowerCase()] || esc(docType);
  const cls   = map[docType.toLowerCase()] ? docType.toLowerCase().replace('é','e').replace('ê','e') : 'texte';
  return `<span class="badge badge--${cls}">${label}</span>`;
}

/**
 * Construit une carte de source (passage retrouvé).
 * @param {object} passage — objet passage de SearchResponse
 * @param {string} currentStrategy
 * @returns {HTMLElement}
 */
export function buildSourceCard(passage, currentStrategy) {
  const card = document.createElement('article');
  card.className = 'source-card';

  // Header
  const header = document.createElement('div');
  header.className = 'source-card__header';

  if (passage.citation_id) {
    const cite = document.createElement('span');
    cite.className = 'source-card__citation';
    cite.textContent = passage.citation_id;
    header.appendChild(cite);
  }

  const titleEl = document.createElement('span');
  titleEl.className = 'source-card__title';
  titleEl.textContent = passage.title || passage.document_id || '—';
  header.appendChild(titleEl);
  card.appendChild(header);

  // Badges métadonnées
  const meta = document.createElement('div');
  meta.className = 'source-card__meta';
  // Libellé lisible de la relation
  const relationLabel = {
    'explicit_reference': 'Référence explicite',
    'parent_context': 'Contexte parent',
    'semantic_similarity': 'Similitude sémantique',
  }[passage.relation] || passage.relation || '';
  meta.innerHTML = [
    passage.document_type ? docTypeBadge(passage.document_type) : '',
    passage.article_number
      ? `<span class="badge badge--article">Art. ${esc(passage.article_number)}</span>` : '',
    passage.page_number
      ? `<span class="badge badge--score">p. ${esc(passage.page_number)}</span>` : '',
    passage.exact_reference
      ? `<span class="badge badge--exact">Référence exacte</span>` : '',
    relationLabel
      ? `<span class="badge badge--score">${esc(relationLabel)}</span>` : '',
    passage.score != null
      ? `<span class="badge badge--score" title="Score de classement par fusion RRF (non normalisé). Plus élevé = mieux classé.">Classement ${passage.score.toFixed(4)}</span>` : '',
  ].join('');
  if (meta.innerHTML) card.appendChild(meta);

  // Extrait
  if (passage.text) {
    const excerpt = document.createElement('div');
    excerpt.className = 'source-card__excerpt';
    excerpt.textContent = passage.text;
    excerpt.title = 'Cliquer pour développer';
    excerpt.addEventListener('click', () => excerpt.classList.toggle('expanded'));
    card.appendChild(excerpt);
  }

  // Références juridiques
  if (passage.references && passage.references.length > 0) {
    const refs = document.createElement('div');
    refs.className = 'source-card__refs';
    passage.references.slice(0, 5).forEach(ref => {
      const chip = document.createElement('span');
      chip.className = 'ref-chip';
      chip.textContent = ref;
      refs.appendChild(chip);
    });
    card.appendChild(refs);
  }

  return card;
}

/**
 * Construit un résumé cohérent des sources (compteurs calculés depuis les
 * mêmes tableaux que les onglets affichés).
 *
 * - Passages : chunk_ids uniques dans le tableau passages (après filtrage/dédup)
 * - Documents : document_ids distincts dans passages uniquement
 * - Références : références juridiques distinctes dans passages uniquement
 * Ne compte pas les catégories dérivées (circulaires, articles, textes_associes)
 * pour éviter le double-comptage signalé (24/19/39 vs 5 sources affichées).
 *
 * @param {object} searchResponse
 * @returns {HTMLElement|null}
 */
export function buildSourcesSummary(searchResponse) {
  const passages = searchResponse.passages || [];
  if (passages.length === 0) return null;

  // Compteurs basés UNIQUEMENT sur les passages — les catégories dérivées
  // (circulaires, articles, textes_associes) sont des sous-ensembles ou
  // des vues croisées des passages ; les inclure crée du double-comptage.
  const chunkIds = new Set(passages.map(p => p.chunk_id).filter(Boolean));
  const docIds   = new Set(passages.map(p => p.document_id).filter(Boolean));
  const refs     = new Set(
    passages.flatMap(p => Array.isArray(p.references) ? p.references : []).filter(Boolean)
  );

  const el = document.createElement('div');
  el.style.cssText = 'font-size:var(--text-xs);color:var(--color-text-muted);'
                   + 'padding:var(--space-xs) var(--space-sm);margin-bottom:var(--space-sm);'
                   + 'background:#f8f9fc;border-radius:var(--radius-sm);'
                   + 'border-left:3px solid var(--color-primary)';
  el.innerHTML = `<strong>${chunkIds.size}</strong> passage${chunkIds.size > 1 ? 's' : ''} · `
              + `<strong>${docIds.size}</strong> document${docIds.size > 1 ? 's' : ''} · `
              + `<strong>${refs.size}</strong> référence${refs.size > 1 ? 's juridiques' : ' juridique'}`;
  return el;
}

/**
 * Construit le bloc de sources complet avec onglets.
 * @param {object} searchResponse — réponse complète de SearchResponse.to_dict()
 * @param {string} currentStrategy
 * @returns {HTMLElement|null}
 */
export function buildSourcesSection(searchResponse, currentStrategy) {
  const tabs = [
    { key: 'passages',       label: 'Passages',           items: searchResponse.passages       || [] },
    { key: 'circulaires',    label: 'Circulaires',         items: searchResponse.circulaires    || [] },
    { key: 'textes_associes',label: 'Textes associés',    items: searchResponse.textes_associes|| [] },
    { key: 'articles',       label: 'Articles',            items: searchResponse.articles       || [] },
  ].filter(t => t.items.length > 0);

  if (tabs.length === 0) return null;

  const section = document.createElement('div');
  section.className = 'sources-section';

  // Bouton toggle
  const toggle = document.createElement('button');
  toggle.className = 'sources-section__toggle';
  toggle.innerHTML = `
    <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2.5">
      <path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/>
    </svg>
    Sources (${tabs.reduce((s, t) => s + t.items.length, 0)})`;

  const content = document.createElement('div');
  content.style.display = 'none';

  toggle.addEventListener('click', () => {
    const open = content.style.display !== 'none';
    content.style.display = open ? 'none' : 'block';
    toggle.classList.toggle('open', !open);
  });

  // Onglets
  const tabBar = document.createElement('div');
  tabBar.className = 'sources-tabs';

  const panels = {};
  tabs.forEach((tab, i) => {
    const btn = document.createElement('button');
    btn.className = 'sources-tab' + (i === 0 ? ' active' : '');
    btn.innerHTML = `${esc(tab.label)}<span class="count">${tab.items.length}</span>`;
    btn.dataset.tab = tab.key;
    tabBar.appendChild(btn);

    const panel = document.createElement('div');
    panel.className = 'sources-panel' + (i === 0 ? ' active' : '');
    panel.dataset.panel = tab.key;
    tab.items.forEach(p => panel.appendChild(buildSourceCard(p, currentStrategy)));
    panels[tab.key] = panel;
  });

  tabBar.addEventListener('click', e => {
    const btn = e.target.closest('.sources-tab');
    if (!btn) return;
    const key = btn.dataset.tab;
    tabBar.querySelectorAll('.sources-tab').forEach(b => b.classList.remove('active'));
    content.querySelectorAll('.sources-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    if (panels[key]) panels[key].classList.add('active');
  });

  content.appendChild(tabBar);
  Object.values(panels).forEach(p => content.appendChild(p));

  section.appendChild(toggle);
  section.appendChild(content);
  return section;
}

/**
 * Construit la bulle de chargement animée.
 * @returns {HTMLElement}
 */
export function buildLoadingBubble() {
  const el = document.createElement('div');
  el.className = 'loading-bubble';
  el.innerHTML = `
    <div class="spinner">
      <span></span><span></span><span></span>
    </div>
    <span style="font-size:var(--text-sm);color:var(--color-text-muted)">Recherche en cours…</span>`;
  return el;
}

/**
 * Construit un message d'erreur.
 * @param {string} message
 * @returns {HTMLElement}
 */
export function buildErrorBubble(message) {
  const el = document.createElement('div');
  el.className = 'error-bubble';
  el.innerHTML = `
    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
    </svg>
    ${esc(message)}`;
  return el;
}

/**
 * Construit la liste des avertissements.
 * @param {string[]} warnings
 * @returns {HTMLElement|null}
 */
export function buildWarnings(warnings) {
  if (!warnings || warnings.length === 0) return null;
  const list = document.createElement('div');
  list.className = 'warnings-list';
  warnings.forEach(w => {
    const item = document.createElement('div');
    item.className = 'warning-item';
    item.innerHTML = `
      <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
        <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v2m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
      </svg>
      ${esc(w)}`;
    list.appendChild(item);
  });
  return list;
}

/**
 * Construit un message complet (user ou assistant).
 * @param {'user'|'assistant'} role
 * @param {string|HTMLElement} content
 * @param {object} [meta]  — { strategy, mode, elapsed }
 * @returns {HTMLElement}
 */
export function buildMessage(role, content, meta = {}) {
  const msg = document.createElement('div');
  msg.className = `message message--${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'message__avatar';
  avatar.textContent = role === 'user' ? 'Q' : 'IA';
  msg.appendChild(avatar);

  const body = document.createElement('div');
  body.className = 'message__body';

  const bubble = document.createElement('div');
  bubble.className = 'message__bubble';

  if (typeof content === 'string') {
    const p = document.createElement('p');
    p.className = 'answer-text';
    p.textContent = content;
    bubble.appendChild(p);
  } else if (content instanceof HTMLElement) {
    bubble.appendChild(content);
  }

  body.appendChild(bubble);

  // Méta (stratégie, mode, temps)
  if (role === 'assistant' && (meta.strategy || meta.elapsed != null)) {
    const metaEl = document.createElement('div');
    metaEl.className = 'message__meta';
    const parts = [];
    if (meta.strategy)          parts.push(esc(meta.strategy));
    if (meta.retrieval_mode)    parts.push(esc(meta.retrieval_mode));
    if (meta.elapsed != null)   parts.push(`${meta.elapsed.toFixed(1)}s`);
    metaEl.textContent = parts.join(' · ');
    body.appendChild(metaEl);
  }

  msg.appendChild(body);
  return msg;
}
