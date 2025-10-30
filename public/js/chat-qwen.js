/**
 * QWEN Chat Module
 * Handles chat messaging, API communication, and UI rendering
 */

class ChatManager {
    constructor() {
        this.messages = [];
        this.isLoading = false;
        this.autoScroll = true;
        this.currentSources = [];
        this.maxResults = 5;
    }

    /**
     * Add a message to the chat
     */
    addMessage(role, content, sources = null) {
        const message = {
            id: Date.now(),
            role,
            content,
            sources,
            timestamp: new Date()
        };
        this.messages.push(message);
        return message;
    }

    /**
     * Clear all messages (new chat)
     */
    clearMessages() {
        this.messages = [];
        this.currentSources = [];
    }

    /**
     * Render a single message to the DOM
     */
    renderMessage(message) {
        const container = document.getElementById('messagesContainer');
        const messageEl = document.createElement('div');
        messageEl.className = `message ${message.role}`;
        messageEl.id = `msg-${message.id}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        // Render markdown-style content
        let html = this.markdownToHtml(message.content);
        bubble.innerHTML = html;

        messageEl.appendChild(bubble);

        // Add sources section if present
        if (message.sources && message.sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'message-sources';

            const sourcesTags = message.sources
                .map((src, idx) => `<span class="sources-tag" data-idx="${idx}">[${idx + 1}]</span>`)
                .join(' ');

            sourcesDiv.innerHTML = sourcesTags;
            messageEl.appendChild(sourcesDiv);

            // Add click handlers for source tags
            sourcesDiv.querySelectorAll('.sources-tag').forEach(tag => {
                tag.addEventListener('click', (e) => {
                    const idx = parseInt(e.target.dataset.idx);
                    this.showSourcesPanel(message.sources, idx);
                });
            });
        }

        container.appendChild(messageEl);

        // Auto-scroll if enabled
        if (this.autoScroll) {
            this.scrollToBottom();
        }

        return messageEl;
    }

    /**
     * Convert markdown-style text to HTML
     */
    markdownToHtml(text) {
        // Escape HTML
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');

        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Italics
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Code blocks
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

        // Inline code
        html = html.replace(/`(.*?)`/g, '<code>$1</code>');

        // Links
        html = html.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Line breaks
        html = html.replace(/\n/g, '<br>');

        return html;
    }

    /**
     * Scroll to the bottom of the messages container
     */
    scrollToBottom() {
        const container = document.getElementById('messagesContainer');
        if (container) {
            setTimeout(() => {
                container.scrollTop = container.scrollHeight;
            }, 0);
        }
    }

    /**
     * Show the sources panel
     */
    showSourcesPanel(sources, highlightIdx = null) {
        const panel = document.getElementById('sourcePanel');
        const sourcesList = document.getElementById('sourcesList');

        sourcesList.innerHTML = '';

        sources.forEach((source, idx) => {
            const sourceEl = document.createElement('div');
            sourceEl.className = 'source-item';
            if (idx === highlightIdx) {
                sourceEl.style.borderColor = 'var(--primary)';
                sourceEl.style.backgroundColor = 'rgba(0, 102, 204, 0.05)';
            }

            const title = document.createElement('div');
            title.className = 'source-title';
            title.textContent = source.title || `Source ${idx + 1}`;

            const snippet = document.createElement('div');
            snippet.className = 'source-snippet';
            snippet.textContent = source.text?.substring(0, 150) + '...' || source.content?.substring(0, 150) + '...';

            const confidence = document.createElement('div');
            confidence.className = 'source-confidence';
            confidence.textContent = `Score: ${(source.score || 0).toFixed(3)}`;

            sourceEl.appendChild(title);
            sourceEl.appendChild(snippet);
            sourceEl.appendChild(confidence);

            sourceEl.addEventListener('click', () => {
                // Copy source text to clipboard or open in new tab
                if (source.url) {
                    window.open(source.url, '_blank');
                }
            });

            sourcesList.appendChild(sourceEl);
        });

        panel.style.display = 'flex';
    }

    /**
     * Hide the sources panel
     */
    hideSourcesPanel() {
        const panel = document.getElementById('sourcePanel');
        panel.style.display = 'none';
    }

    /**
     * Start loading indicator
     */
    startLoading() {
        this.isLoading = true;
        const container = document.getElementById('messagesContainer');
        const loadingEl = document.createElement('div');
        loadingEl.className = 'message assistant';
        loadingEl.id = 'loading-message';
        loadingEl.innerHTML = `
            <div class="message-bubble">
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
            </div>
        `;
        container.appendChild(loadingEl);
        this.scrollToBottom();
    }

    /**
     * Stop loading indicator
     */
    stopLoading() {
        this.isLoading = false;
        const loadingEl = document.getElementById('loading-message');
        if (loadingEl) {
            loadingEl.remove();
        }
    }

    /**
     * Set loading state for UI
     */
    setLoadingState(isLoading) {
        const sendBtn = document.getElementById('sendBtn');
        const chatInput = document.getElementById('chatInput');

        if (isLoading) {
            sendBtn.disabled = true;
            sendBtn.classList.add('disabled');
            chatInput.disabled = true;
        } else {
            sendBtn.disabled = false;
            sendBtn.classList.remove('disabled');
            chatInput.disabled = false;
            chatInput.focus();
        }
    }

    /**
     * Handle error message
     */
    showError(error) {
        const errorMsg = error.message || 'An error occurred. Please try again.';
        const message = this.addMessage('assistant', `❌ ${errorMsg}`);
        this.renderMessage(message);
    }

    /**
     * Update settings
     */
    updateSettings(settings) {
        if ('autoScroll' in settings) {
            this.autoScroll = settings.autoScroll;
        }
        if ('maxResults' in settings) {
            this.maxResults = settings.maxResults;
        }
    }
}

// Create global chat manager instance
const chatManager = new ChatManager();
