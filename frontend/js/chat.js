/**
 * chat.js — Logique de la page de conversation.
 */

import { search, ApiError } from './api.js';
import { CONFIG } from './config.js';
import { addToHistory, getHistory, removeFromHistory, relativeTime } from './store.js';
import {
  buildMessage, buildLoadingBubble, buildErrorBubble,
  buildSourcesSection, buildSourcesSummary, buildWarnings,
} from './components.js';

// ── Références DOM ──────────────────────────────────────────────────────────
const messagesEl   = document.getElementById('chat-messages');
const inputEl      = document.getElementById('chat-input');
const sendBtn      = document.getElementById('btn-send');
const strategyEl   = document.getElementById('chat-strategy');
const modeEl       = document.getElementById('chat-mode');
const limitEl      = document.getElementById('chat-limit');
const sidebarList  = document.getElementById('sidebar-list');
const welcomeEl    = document.getElementById('welcome-screen');

// ── Initialisation ──────────────────────────────────────────────────────────
export function initChat() {
  renderSidebar();
  setupInput();
  setupWelcomeChips();
}

// ── Sidebar historique ───────────────────────────────────────────────────────
function renderSidebar() {
  if (!sidebarList) return;
  const history = getHistory();
  sidebarList.innerHTML = '';

  if (history.length === 0) {
    sidebarList.innerHTML = `
      <div class="sidebar__empty">
        <svg width="32" height="32" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
          <path stroke-linecap="round" stroke-linejoin="round"
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
        </svg>
        Aucune conversation
      </div>`;
    return;
  }

  history.forEach(item => {
    const div = document.createElement('div');
    div.className = 'chat-item';

    // Zone de texte cliquable (réutilise la question dans l'input)
    const textArea = document.createElement('div');
    textArea.className = 'chat-item__text';
    textArea.innerHTML = `
      <div class="chat-item__icon">
        <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
          <path stroke-linecap="round" stroke-linejoin="round"
            d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"/>
        </svg>
      </div>
      <div>
        <div class="chat-item__query">${escHtml(item.query)}</div>
        <div class="chat-item__meta">${relativeTime(item.ts)}</div>
      </div>`;
    textArea.style.cssText = 'display:flex;align-items:flex-start;gap:var(--space-sm);flex:1;min-width:0;cursor:pointer';
    textArea.addEventListener('click', () => {
      if (inputEl) {
        inputEl.value = item.query;
        inputEl.focus();
        adjustHeight(inputEl);
        sendBtn.disabled = false;
      }
    });

    // Bouton de suppression
    const delBtn = document.createElement('button');
    delBtn.className = 'chat-item__delete';
    delBtn.title = 'Supprimer cette conversation';
    delBtn.setAttribute('aria-label', 'Supprimer');
    delBtn.innerHTML = `
      <svg width="13" height="13" fill="none" viewBox="0 0 24 24"
           stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="3 6 5 6 21 6"/>
        <path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>
        <path d="M10 11v6M14 11v6"/>
        <path d="M9 6V4a1 1 0 011-1h4a1 1 0 011 1v2"/>
      </svg>`;
    delBtn.addEventListener('click', e => {
      e.stopPropagation();   // ne pas déclencher le clic sur textArea
      removeFromHistory(item.id);
      renderSidebar();
    });

    div.appendChild(textArea);
    div.appendChild(delBtn);
    sidebarList.appendChild(div);
  });
}

// ── Input ────────────────────────────────────────────────────────────────────
function setupInput() {
  if (!inputEl || !sendBtn) return;

  inputEl.addEventListener('input', () => {
    adjustHeight(inputEl);
    sendBtn.disabled = !inputEl.value.trim();
  });

  inputEl.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) handleSend();
    }
  });

  sendBtn.addEventListener('click', handleSend);
  sendBtn.disabled = true;
}

function adjustHeight(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

// ── Welcome chips ────────────────────────────────────────────────────────────
function setupWelcomeChips() {
  document.querySelectorAll('.welcome__chip').forEach(chip => {
    chip.addEventListener('click', () => {
      if (inputEl) {
        inputEl.value = chip.textContent.trim();
        adjustHeight(inputEl);
        sendBtn.disabled = false;
        inputEl.focus();
      }
    });
  });
}

// ── Send ─────────────────────────────────────────────────────────────────────
async function handleSend() {
  const query = inputEl.value.trim();
  if (!query) return;

  // Cacher l'écran de bienvenue
  if (welcomeEl) welcomeEl.style.display = 'none';

  // Message utilisateur
  appendMessage(buildMessage('user', query));

  // Reset input
  inputEl.value = '';
  adjustHeight(inputEl);
  sendBtn.disabled = true;

  // Paramètres
  const strategy       = strategyEl?.value  || 'structural';
  const retrieval_mode = modeEl?.value      || 'hybrid';
  const limit          = parseInt(limitEl?.value || '10', 10);

  // Bulle de chargement
  const loadingMsg = buildAssistantFrame();
  const loadingBubble = buildLoadingBubble();
  loadingMsg.querySelector('.message__bubble').appendChild(loadingBubble);
  appendMessage(loadingMsg);

  const t0 = performance.now();
  try {
    // generate_answer=true : le serveur (server.py) branche FaithfulAnswerGenerator.
    // Sans générateur configuré, le backend retourne answer=null + avertissement.
    const response = await search({ query, strategy, retrieval_mode, limit, generate_answer: true });
    const elapsed = (performance.now() - t0) / 1000;

    // Remplacer la bulle de chargement
    loadingMsg.remove();

    // Construire le contenu de la réponse
    const bodyEl = document.createElement('div');

    // ── Réponse générée (synthèse fidèle ou réponse insuffisante) ──────────
    if (response.answer) {
      const isInsufficient = response.answer.includes('ne permet pas d\'établir une réponse fiable');
      if (isInsufficient) {
        // Cas "corpus insuffisant" : affiché comme avertissement distinct, pas comme synthèse.
        const warn = document.createElement('div');
        warn.className = 'warning-item';
        warn.style.cssText += 'margin:0 0 var(--space-md) 0;display:block';
        const label = document.createElement('p');
        label.style.cssText = 'font-size:var(--text-xs);font-weight:700;color:var(--color-warning);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px';
        label.textContent = 'Réponse insuffisante';
        warn.appendChild(label);
        const text = document.createElement('p');
        text.className = 'answer-text';
        text.textContent = response.answer;
        warn.appendChild(text);
        bodyEl.appendChild(warn);
      } else {
        // Synthèse générée à partir des passages retrouvés.
        const answerEl = document.createElement('div');
        const label = document.createElement('p');
        label.style.cssText = 'font-size:var(--text-xs);font-weight:700;color:var(--color-text-muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px';
        label.textContent = 'Synthèse générée à partir des sources';
        answerEl.appendChild(label);
        const text = document.createElement('p');
        text.className = 'answer-text';
        text.textContent = response.answer;
        answerEl.appendChild(text);
        bodyEl.appendChild(answerEl);
      }
    } else {
      // Pas de générateur configuré côté serveur — état neutre.
      const p = document.createElement('p');
      p.className = 'answer-text';
      p.style.color = 'var(--color-text-muted)';
      const total = (response.passages || []).length;
      if (total > 0) {
        p.textContent = `${total} passage${total > 1 ? 's' : ''} retrouvé${total > 1 ? 's' : ''}. Consultez les sources ci-dessous.`;
      } else {
        p.textContent = 'Aucun passage pertinent trouvé dans les documents indexés.';
      }
      bodyEl.appendChild(p);
    }

    // ── Résumé des sources (compteurs cohérents avec les onglets affichés) ─
    const summary = buildSourcesSummary(response);
    if (summary) bodyEl.appendChild(summary);

    // Avertissements du backend
    const warnEl = buildWarnings(response.warnings);
    if (warnEl) bodyEl.appendChild(warnEl);

    // Sources détaillées (onglets)
    const sourcesEl = buildSourcesSection(response, strategy);
    if (sourcesEl) bodyEl.appendChild(sourcesEl);

    const msg = buildMessage('assistant', bodyEl, {
      strategy,
      retrieval_mode,
      elapsed,
    });
    appendMessage(msg);

    // Historique
    addToHistory(query);
    renderSidebar();

  } catch (err) {
    loadingMsg.remove();
    const errContent = document.createElement('div');
    errContent.appendChild(buildErrorBubble(
      err instanceof ApiError
        ? `${err.message}${err.detail ? ` — ${err.detail}` : ''}`
        : 'Une erreur inattendue s\'est produite.'
    ));
    appendMessage(buildMessage('assistant', errContent));
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function buildAssistantFrame() {
  const msg = document.createElement('div');
  msg.className = 'message message--assistant';
  msg.innerHTML = `
    <div class="message__avatar">IA</div>
    <div class="message__body">
      <div class="message__bubble"></div>
    </div>`;
  return msg;
}

function appendMessage(el) {
  if (!messagesEl) return;
  messagesEl.appendChild(el);
  el.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
