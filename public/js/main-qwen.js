/**
 * QWEN Chat UI - Main JavaScript
 * Handles initialization, event listeners, and API communication
 */

// ===== API Configuration =====
const API_TOKEN = localStorage.getItem('api_token') || 'change-me';
let API_BASE = (function(){
    if (window.__API_BASE__) return window.__API_BASE__;
    const loc = window.location;
    if (loc.port && loc.port !== '7001') {
        return `${loc.protocol}//${loc.hostname}:7001`;
    }
    return loc.origin;
})();

// ===== DOM Elements =====
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const settingsBtn = document.getElementById('settingsBtn');
const infoBtn = document.getElementById('infoBtn');
const closeSourceBtn = document.getElementById('closeSourceBtn');
const messagesContainer = document.getElementById('messagesContainer');
const emptyState = document.getElementById('emptyState');
const sourcePanel = document.getElementById('sourcePanel');
const settingsModal = document.getElementById('settingsModal');
const infoModal = document.getElementById('infoModal');

// ===== Settings Modal Controls =====
const closeSettingsBtn = document.getElementById('closeSettingsBtn');
const closeSettingsBtn2 = document.getElementById('closeSettingsBtn2');
const closeInfoBtn = document.getElementById('closeInfoBtn');

// ===== Settings Input Elements =====
const autoScrollCheckbox = document.getElementById('autoScroll');
const darkModeCheckbox = document.getElementById('darkMode');
const showSourcesCheckbox = document.getElementById('showSources');
const maxResultsInput = document.getElementById('maxResults');

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
    initializeUI();
    loadSettings();
    setupEventListeners();
    showWelcomeMessage();
});

// ===== Initialization =====
function initializeUI() {
    // Check for saved dark mode preference
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        darkModeCheckbox.checked = true;
    }

    // Set focus on input
    chatInput.focus();
}

function loadSettings() {
    const settings = JSON.parse(localStorage.getItem('chatSettings') || '{}');

    if ('autoScroll' in settings) {
        autoScrollCheckbox.checked = settings.autoScroll;
        chatManager.autoScroll = settings.autoScroll;
    }

    if ('showSources' in settings) {
        showSourcesCheckbox.checked = settings.showSources;
    }

    if ('maxResults' in settings) {
        maxResultsInput.value = settings.maxResults;
        chatManager.maxResults = settings.maxResults;
    }
}

function saveSettings() {
    const settings = {
        autoScroll: autoScrollCheckbox.checked,
        showSources: showSourcesCheckbox.checked,
        maxResults: parseInt(maxResultsInput.value)
    };
    localStorage.setItem('chatSettings', JSON.stringify(settings));
    chatManager.updateSettings(settings);
}

// ===== Event Listeners =====
function setupEventListeners() {
    // Chat input
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);
    newChatBtn.addEventListener('click', startNewChat);
    settingsBtn.addEventListener('click', openSettings);
    infoBtn.addEventListener('click', openInfo);
    closeSourceBtn.addEventListener('click', () => sourcePanel.style.display = 'none');

    // Settings modal
    closeSettingsBtn.addEventListener('click', closeSettings);
    closeSettingsBtn2.addEventListener('click', closeSettings);
    closeInfoBtn.addEventListener('click', closeInfo);

    // Settings changes
    autoScrollCheckbox.addEventListener('change', saveSettings);
    showSourcesCheckbox.addEventListener('change', saveSettings);
    maxResultsInput.addEventListener('change', saveSettings);

    // Dark mode toggle
    darkModeCheckbox.addEventListener('change', () => {
        document.body.classList.toggle('dark-mode');
        localStorage.setItem('darkMode', darkModeCheckbox.checked);
    });

    // Auto-resize textarea
    chatInput.addEventListener('input', autoResizeTextarea);
}

function autoResizeTextarea() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
}

// ===== Chat Functions =====
async function sendMessage() {
    const message = chatInput.value.trim();

    if (!message || chatManager.isLoading) {
        return;
    }

    // Hide empty state
    emptyState.style.display = 'none';

    // Add user message
    const userMsg = chatManager.addMessage('user', message);
    chatManager.renderMessage(userMsg);

    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Show loading
    chatManager.startLoading();
    chatManager.setLoadingState(true);

    try {
        // Call API
        const response = await callChatAPI(message);

        // Stop loading
        chatManager.stopLoading();

        // Add assistant message
        const assistantMsg = chatManager.addMessage(
            'assistant',
            response.response,
            response.sources || []
        );
        chatManager.renderMessage(assistantMsg);

        // Show sources panel if requested
        if (showSourcesCheckbox.checked && response.sources && response.sources.length > 0) {
            chatManager.showSourcesPanel(response.sources);
        }

    } catch (error) {
        console.error('Chat error:', error);
        chatManager.stopLoading();
        chatManager.showError(error);
    } finally {
        chatManager.setLoadingState(false);
        chatInput.focus();
    }
}

async function callChatAPI(question) {
    const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'x-api-token': API_TOKEN
        },
        body: JSON.stringify({
            question,
            namespace: 'clockify',
            k: chatManager.maxResults
        })
    });

    if (!response.ok) {
        throw new Error(`API error: ${response.statusText}`);
    }

    const data = await response.json();
    return {
        response: data.response,
        sources: data.sources || []
    };
}

function startNewChat() {
    if (confirm('Start a new chat? Current conversation will be cleared.')) {
        chatManager.clearMessages();
        messagesContainer.innerHTML = '';
        emptyState.style.display = 'flex';
        sourcePanel.style.display = 'none';
        chatInput.value = '';
        chatInput.focus();
    }
}

function showWelcomeMessage() {
    // Don't show welcome if there are existing messages
    if (chatManager.messages.length === 0) {
        // The empty state is shown by default in HTML
    }
}

// ===== Modal Functions =====
function openSettings() {
    settingsModal.style.display = 'flex';
}

function closeSettings() {
    settingsModal.style.display = 'none';
}

function openInfo() {
    infoModal.style.display = 'flex';
}

function closeInfo() {
    infoModal.style.display = 'none';
}

// ===== Modal Click Outside to Close =====
window.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
        closeSettings();
    }
    if (e.target === infoModal) {
        closeInfo();
    }
});

// ===== Keyboard Shortcuts =====
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K for new chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        startNewChat();
    }

    // Ctrl/Cmd + , for settings
    if ((e.ctrlKey || e.metaKey) && e.key === ',') {
        e.preventDefault();
        openSettings();
    }
});

// ===== Utility Functions =====
function getApiToken() {
    return localStorage.getItem('api_token') || 'change-me';
}

function setApiToken(token) {
    localStorage.setItem('api_token', token);
}

// ===== Page Visibility =====
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        chatInput.focus();
    }
});

console.log('🚀 Clockify RAG Chat initialized');
