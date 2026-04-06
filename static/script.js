const MAX_HISTORY = 80; // máx de mensagens por chat

const messagesEl   = document.getElementById('messages');
const inputEl      = document.getElementById('user-input');
const sendBtn      = document.getElementById('send-btn');
const chatListEl   = document.getElementById('chat-list');
const newChatBtn   = document.getElementById('new-chat-btn');
const chatTitleEl  = document.getElementById('chat-title');
const sidebarEmpty = document.getElementById('sidebar-empty');

// chats = { [id]: { title: string, messages: [{role, text}] } }
const chats = {};
let currentChatId = null;

function generateId() {
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

function formatResponse(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/^-{3,}$/gm, '<hr>')
    .replace(/^[-•]\s*(.+)/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    .replace(/\n/g, '<br>');
}

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/\n/g, '<br>');
}

function escapeAttr(text) {
  return String(text).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function renderSidebar() {
  Array.from(chatListEl.querySelectorAll('.chat-item')).forEach(el => el.remove());

  const ids = Object.keys(chats).reverse(); // mais recente primeiro

  if (ids.length === 0) {
    sidebarEmpty.style.display = 'block';
    return;
  }
  sidebarEmpty.style.display = 'none';

  ids.forEach(id => {
    const chat = chats[id];
    const item = document.createElement('div');
    item.className = 'chat-item' + (id === currentChatId ? ' active' : '');
    item.dataset.id = id;

    item.innerHTML = `
      <span class="chat-item-icon">💬</span>
      <span class="chat-item-title" title="${escapeAttr(chat.title)}">${escapeHtml(chat.title)}</span>
      <button class="chat-item-delete" title="Excluir conversa" aria-label="Excluir">✕</button>
    `;

    item.addEventListener('click', (e) => {
      if (e.target.closest('.chat-item-delete')) return;
      switchToChat(id);
    });

    item.querySelector('.chat-item-delete').addEventListener('click', (e) => {
      e.stopPropagation();
      deleteChat(id);
    });

    chatListEl.appendChild(item);
  });
}

function createNewChat() {
  const id = generateId();
  chats[id] = { title: 'Nova conversa', messages: [] };
  switchToChat(id);
}

function switchToChat(id) {
  currentChatId = id;
  renderSidebar();
  renderMessages();
  inputEl.focus();
}

function deleteChat(id) {
  const title = chats[id]?.title || 'esta conversa';
  if (!confirm(`Excluir "${title}"?`)) return;

  fetch('/chat/reset', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ chat_id: id }),
  }).catch(() => {});

  delete chats[id];

  if (id === currentChatId) {
    const remaining = Object.keys(chats);
    currentChatId = remaining.length > 0 ? remaining[remaining.length - 1] : null;
  }

  renderSidebar();
  renderMessages();
}

function renderMessages() {
  messagesEl.innerHTML = '';

  if (!currentChatId || !chats[currentChatId]) {
    chatTitleEl.textContent = 'ChefBot';
    messagesEl.appendChild(buildEmptyState());
    return;
  }

  const chat = chats[currentChatId];
  chatTitleEl.textContent = chat.title;

  if (chat.messages.length === 0) {
    messagesEl.appendChild(buildEmptyState());
    return;
  }

  const fragment = document.createDocumentFragment();
  chat.messages.forEach(msg => {
    const html = msg.role === 'bot' ? formatResponse(msg.text) : escapeHtml(msg.text);
    const row  = buildBubble(msg.role, html);
    row.style.animation = 'none';
    row.style.opacity   = '1';
    fragment.appendChild(row);
  });
  messagesEl.appendChild(fragment);
  scrollToBottom(false);
}

function buildEmptyState() {
  const div = document.createElement('div');
  div.className = 'empty-state';
  div.innerHTML = `
    <div class="icon">👨‍🍳</div>
    <h2>Olá! Sou o ChefBot.</h2>
    <p>Me diga quais ingredientes você tem na geladeira ou me peça uma receita especial.</p>
  `;
  return div;
}

function buildBubble(role, html) {
  const row = document.createElement('div');
  row.className = `msg-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${role}`;
  avatar.textContent = role === 'bot' ? '🍳' : '👤';

  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  bubble.innerHTML = html;

  row.appendChild(avatar);
  row.appendChild(bubble);
  return row;
}

function addMessageToDOM(role, html) {
  if (role === 'bot') removeTypingIndicator();
  messagesEl.querySelector('.empty-state')?.remove();

  const row = buildBubble(role, html);
  messagesEl.appendChild(row);
  scrollToBottom();
}

function persistMessage(role, text) {
  if (!currentChatId || !chats[currentChatId]) return;

  const msgs = chats[currentChatId].messages;
  msgs.push({ role, text });

  if (role === 'user' && msgs.filter(m => m.role === 'user').length === 1) {
    chats[currentChatId].title = text.slice(0, 42) + (text.length > 42 ? '…' : '');
    chatTitleEl.textContent = chats[currentChatId].title;
    renderSidebar();
  }

  if (msgs.length > MAX_HISTORY) {
    chats[currentChatId].messages = msgs.slice(-MAX_HISTORY);
  }
}

async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed) return;

  if (!currentChatId) createNewChat();

  addMessageToDOM('user', escapeHtml(trimmed));
  persistMessage('user', trimmed);

  inputEl.value = '';
  autoResize();
  setLoading(true);
  showTypingIndicator();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mensagem: trimmed, chat_id: currentChatId }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const data     = await res.json();
    const resposta = data.resposta || 'Desculpe, não consegui processar sua mensagem.';

    addMessageToDOM('bot', formatResponse(resposta));
    persistMessage('bot', resposta);

  } catch (err) {
    addMessageToDOM('bot', escapeHtml('⚠️ Não foi possível conectar ao servidor. Verifique se o Flask está rodando.'));
    console.error('[ChefBot]', err);
  } finally {
    setLoading(false);
  }
}

function scrollToBottom(smooth = true) {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: smooth ? 'smooth' : 'instant' });
}

function setLoading(loading) {
  sendBtn.disabled  = loading;
  inputEl.disabled  = loading;
  inputEl.style.opacity = loading ? '0.7' : '1';
  if (!loading) inputEl.focus();
}

function showTypingIndicator() {
  const row = buildBubble('bot', `<div class="typing-dots"><span></span><span></span><span></span></div>`);
  row.id = 'typing-row';
  messagesEl.appendChild(row);
  scrollToBottom();
}

function removeTypingIndicator() {
  document.getElementById('typing-row')?.remove();
}

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
}

newChatBtn.addEventListener('click', createNewChat);

sendBtn.addEventListener('click', () => sendMessage(inputEl.value));

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage(inputEl.value);
  }
});

inputEl.addEventListener('input', autoResize);

function sendSuggestion(text) {
  inputEl.value = text;
  autoResize();
  setTimeout(() => sendMessage(text), 150);
}

window.sendSuggestion = sendSuggestion;

renderSidebar();
renderMessages();
inputEl.focus();