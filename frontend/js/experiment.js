/**
 * experiment.js — Page de comparaison des stratégies de recherche.
 *
 * L'évaluation par métriques (P@k, R@k, MRR, nDCG) n'est pas disponible
 * via l'API du backend (/api/search ne retourne pas les métriques calculées).
 * La page affiche donc les passages retournés et leur classement, et indique
 * clairement ce qui manque côté backend pour calculer les métriques.
 */

import { search, ApiError } from './api.js';
import { CONFIG } from './config.js';
import { buildSourceCard, docTypeBadge } from './components.js';

// ── Références DOM ──────────────────────────────────────────────────────────
const expQueryEl  = document.getElementById('exp-query');
const expModeEl   = document.getElementById('exp-mode');
const expLimitEl  = document.getElementById('exp-limit');
const runBtn      = document.getElementById('btn-run-exp');
const compareBtn  = document.getElementById('btn-compare-all');
const resultsArea = document.getElementById('exp-results');
const metricsArea = document.getElementById('exp-metrics');

// ── Init ────────────────────────────────────────────────────────────────────
export function initExperiment() {
  if (!runBtn) return;
  runBtn.addEventListener('click', () => runExperiment(false));
  if (compareBtn) compareBtn.addEventListener('click', () => runExperiment(true));

  // Exemples de questions
  document.querySelectorAll('.exp-question-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      if (expQueryEl) {
        expQueryEl.value = chip.textContent.trim();
        expQueryEl.focus();
      }
    });
  });
}

// ── Lancement ────────────────────────────────────────────────────────────────
async function runExperiment(allStrategies) {
  const query = expQueryEl?.value.trim();
  if (!query) {
    expQueryEl?.focus();
    return;
  }

  const mode    = expModeEl?.value  || 'semantic';
  const limit   = parseInt(expLimitEl?.value || '5', 10);

  const strategies = allStrategies
    ? CONFIG.STRATEGIES.map(s => s.value)
    : [document.getElementById('exp-strategy')?.value || 'structural'];

  // Désactiver les boutons
  setRunning(true);

  // Vider et créer les cartes vides
  resultsArea.innerHTML = '';
  const cards = {};
  strategies.forEach(strategy => {
    const card = buildResultCard(strategy, mode);
    resultsArea.appendChild(card.el);
    cards[strategy] = card;
  });

  // Lancer toutes les requêtes en parallèle
  const results = {};
  await Promise.allSettled(
    strategies.map(async strategy => {
      const t0 = performance.now();
      try {
        const resp = await search({ query, strategy, retrieval_mode: mode, limit });
        const elapsed = (performance.now() - t0) / 1000;
        results[strategy] = { resp, elapsed };
        fillResultCard(cards[strategy], resp, elapsed, strategy);
      } catch (err) {
        const elapsed = (performance.now() - t0) / 1000;
        showCardError(cards[strategy], err, elapsed);
      }
    })
  );

  // Section métriques
  renderMetricsSection(results, strategies);

  setRunning(false);
}

// ── Cartes de résultats ───────────────────────────────────────────────────────
function buildResultCard(strategy, mode) {
  const strategyInfo = CONFIG.STRATEGIES.find(s => s.value === strategy) || { label: strategy };
  const modeInfo     = CONFIG.RETRIEVAL_MODES.find(m => m.value === mode) || { label: mode };

  const el = document.createElement('div');
  el.className = 'exp-result-card';
  el.innerHTML = `
    <div class="exp-result-card__header">
      <span class="exp-result-card__title">${escHtml(strategyInfo.label)} — ${escHtml(modeInfo.label)}</span>
      <span class="exp-result-card__time" data-time></span>
    </div>
    <div class="exp-result-card__body">
      <div class="exp-result-card__loading">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" stroke-dasharray="60" stroke-dashoffset="20">
            <animateTransform attributeName="transform" type="rotate" from="0 12 12" to="360 12 12" dur=".8s" repeatCount="indefinite"/>
          </circle>
        </svg>
        Recherche…
      </div>
    </div>`;
  return { el, strategy, mode };
}

function fillResultCard(card, response, elapsed, strategy) {
  const header = card.el.querySelector('.exp-result-card__header');
  const timeEl = card.el.querySelector('[data-time]');
  const body   = card.el.querySelector('.exp-result-card__body');

  if (timeEl) timeEl.textContent = `${elapsed.toFixed(1)}s`;
  body.innerHTML = '';

  const passages = response.passages || [];
  if (passages.length === 0) {
    body.innerHTML = `<p style="color:var(--color-text-muted);font-size:var(--text-sm);padding:var(--space-md)">Aucun passage retrouvé.</p>`;
    return;
  }

  passages.forEach((p, i) => {
    const div = document.createElement('div');
    div.className = 'exp-passage';
    div.innerHTML = `
      <div class="exp-passage__rank">#${i + 1}${p.citation_id ? ` ${escHtml(p.citation_id)}` : ''}</div>
      <div class="exp-passage__meta">
        ${p.document_type ? docTypeBadge(p.document_type) : ''}
        ${p.article_number ? `<span class="badge badge--article">Art. ${escHtml(p.article_number)}</span>` : ''}
        ${p.score != null ? `<span class="badge badge--score">${p.score.toFixed(4)}</span>` : ''}
        ${p.exact_reference ? `<span class="badge badge--exact">Réf. exacte</span>` : ''}
      </div>
      <div class="exp-passage__meta" style="margin-bottom:4px">
        <span style="font-size:var(--text-xs);color:var(--color-text-muted)">${escHtml(p.title || p.document_id || '')}</span>
      </div>
      <div class="exp-passage__text">${escHtml(p.text || '')}</div>`;
    body.appendChild(div);
  });

  // Avertissements
  (response.warnings || []).forEach(w => {
    const wdiv = document.createElement('div');
    wdiv.style.cssText = 'font-size:var(--text-xs);color:var(--color-warning);padding:4px var(--space-sm);margin-top:4px';
    wdiv.textContent = `⚠ ${w}`;
    body.appendChild(wdiv);
  });
}

function showCardError(card, err, elapsed) {
  const timeEl = card.el.querySelector('[data-time]');
  const body   = card.el.querySelector('.exp-result-card__body');
  if (timeEl) timeEl.textContent = `${elapsed.toFixed(1)}s`;
  body.innerHTML = `
    <div class="exp-result-card__error">
      ✗ ${escHtml(err instanceof ApiError ? err.message : String(err))}
    </div>`;
}

// ── Section métriques ─────────────────────────────────────────────────────────
function renderMetricsSection(results, strategies) {
  if (!metricsArea) return;
  metricsArea.innerHTML = '';

  const section = document.createElement('div');
  section.className = 'metrics-section';
  section.innerHTML = `<h3>Comparaison des résultats</h3>`;

  // Note sur les métriques
  const note = document.createElement('div');
  note.className = 'metrics-note';
  note.innerHTML = `
    <strong>Métriques P@k, R@k, MRR, nDCG@k non disponibles ici.</strong><br>
    L'endpoint <code>/api/search</code> retourne les passages classés mais pas les métriques calculées.
    Pour calculer P@k/R@k, il faut des jugements de pertinence annotés (<code>evaluation/cases.json</code>)
    et lancer <code>run_evaluation()</code> depuis <code>legal_rag/evaluation.py</code>.
    La comparaison ci-dessous porte sur le classement et les scores retournés par le moteur.`;
  section.appendChild(note);

  // Tableau comparatif
  const table = document.createElement('table');
  table.className = 'metrics-table';
  table.innerHTML = `
    <thead><tr>
      <th>Stratégie</th>
      <th>Passages</th>
      <th>Circulaires</th>
      <th>Textes associés</th>
      <th>Score max</th>
      <th>Réf. exacte</th>
      <th>Avertissements</th>
      <th>Temps</th>
    </tr></thead>
    <tbody id="metrics-tbody"></tbody>`;

  strategies.forEach(strategy => {
    const r = results[strategy];
    const tr = document.createElement('tr');
    if (!r) {
      tr.innerHTML = `<td>${escHtml(strategy)}</td><td colspan="7" style="color:#b91c1c">Erreur</td>`;
    } else {
      const { resp, elapsed } = r;
      const maxScore = Math.max(...(resp.passages || []).map(p => p.score || 0), 0);
      const exactCount = (resp.passages || []).filter(p => p.exact_reference).length;
      tr.innerHTML = `
        <td><strong>${escHtml(strategy)}</strong></td>
        <td class="metric-val">${(resp.passages || []).length}</td>
        <td class="metric-val">${(resp.circulaires || []).length}</td>
        <td class="metric-val">${(resp.textes_associes || []).length}</td>
        <td class="metric-val">${maxScore > 0 ? maxScore.toFixed(4) : '—'}</td>
        <td class="metric-val">${exactCount > 0 ? exactCount : '0'}</td>
        <td class="metric-val">${(resp.warnings || []).length}</td>
        <td class="metric-val">${elapsed.toFixed(1)}s</td>`;
    }
    table.querySelector('#metrics-tbody').appendChild(tr);
  });

  section.appendChild(table);
  metricsArea.appendChild(section);
}

// ── Utilitaires ───────────────────────────────────────────────────────────────
function setRunning(running) {
  if (runBtn)     runBtn.disabled     = running;
  if (compareBtn) compareBtn.disabled = running;
}

function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
