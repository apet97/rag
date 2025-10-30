// Configuration
function getConfig() {
    return {
        apiBase: document.getElementById('apiBase').value || 'http://10.127.0.192:7001',
        apiToken: document.getElementById('apiToken').value || '05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0',
        k: parseInt(document.getElementById('kValue').value) || 5
    };
}

// Tab switching
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

// Config panel toggle
function toggleConfigPanel() {
    const header = document.querySelector('.config-header');
    const content = document.getElementById('configContent');
    header.classList.toggle('expanded');
    content.classList.toggle('expanded');

    // Load config on first open
    if (content.classList.contains('expanded') && !content.dataset.loaded) {
        loadSystemConfiguration();
        content.dataset.loaded = 'true';
    }
}

// Load system configuration from /health endpoint
async function loadSystemConfiguration() {
    const config = getConfig();
    const grid = document.getElementById('configGrid');

    try {
        const response = await fetch(`${config.apiBase}/health?detailed=1`, {
            headers: { 'x-api-token': config.apiToken }
        });

        if (!response.ok) {
            grid.innerHTML = '<div style="grid-column: 1/-1; color: #666;">Could not load configuration</div>';
            return;
        }

        const data = await response.json();
        renderSystemConfiguration(data);

        // Update status indicator
        const statusEl = document.getElementById('configStatus');
        if (data.ok) {
            statusEl.innerHTML = '<span class="status-badge enabled">✓ Healthy</span>';
        } else {
            statusEl.innerHTML = '<span class="status-badge disabled">✗ Unhealthy</span>';
        }
    } catch (err) {
        grid.innerHTML = `<div style="grid-column: 1/-1; color: #999;">Error: ${err.message}</div>`;
    }
}

// Render system configuration
function renderSystemConfiguration(data) {
    const grid = document.getElementById('configGrid');
    let html = '';

    // Namespaces
    if (data.namespaces && data.namespaces.length > 0) {
        html += `
            <div class="config-item-detail">
                <div class="config-label">Namespaces</div>
                <div class="config-value">${data.namespaces.join(', ')}</div>
            </div>
        `;
    }

    // Embedding Backend
    if (data.config && data.config.embeddings_backend) {
        const backend = data.config.embeddings_backend;
        const badgeClass = backend === 'stub' ? 'disabled' : 'enabled';
        const badgeText = backend === 'stub' ? 'STUB (Testing)' : 'Model';
        html += `
            <div class="config-item-detail">
                <div class="config-label">Embedding Backend</div>
                <div class="config-value">
                    <span class="status-badge ${badgeClass}">${badgeText}</span>
                </div>
            </div>
        `;
    }

    // Reranker Status
    if (data.config !== undefined) {
        const disabled = data.config && data.config.rerank_disabled;
        const badgeClass = disabled ? 'disabled' : 'enabled';
        const badgeText = disabled ? 'DISABLED' : 'ENABLED';
        html += `
            <div class="config-item-detail">
                <div class="config-label">Reranker</div>
                <div class="config-value">
                    <span class="status-badge ${badgeClass}">${badgeText}</span>
                </div>
            </div>
        `;
    }

    // LLM Model
    if (data.llm_model) {
        html += `
            <div class="config-item-detail">
                <div class="config-label">LLM Model</div>
                <div class="config-value">${escapeHtml(data.llm_model)}</div>
            </div>
        `;
    }

    // Mode
    if (data.mode) {
        const badgeClass = data.mode === 'mock' ? 'unknown' : 'enabled';
        html += `
            <div class="config-item-detail">
                <div class="config-label">Mode</div>
                <div class="config-value">
                    <span class="status-badge ${badgeClass}">${data.mode.toUpperCase()}</span>
                </div>
            </div>
        `;
    }

    // Cache Stats
    if (data.cache_hit_rate_pct !== undefined) {
        const hitRate = data.cache_hit_rate_pct;
        const badgeClass = hitRate > 50 ? 'enabled' : 'unknown';
        html += `
            <div class="config-item-detail">
                <div class="config-label">Cache Hit Rate</div>
                <div class="config-value">
                    <span style="color: #333;">${hitRate.toFixed(1)}%</span>
                </div>
            </div>
        `;
    }

    // Cache Memory
    if (data.cache_memory_mb !== undefined) {
        html += `
            <div class="config-item-detail">
                <div class="config-label">Cache Memory</div>
                <div class="config-value">${data.cache_memory_mb.toFixed(2)} MB</div>
            </div>
        `;
    }

    // Vector Count
    if (data.index_metrics) {
        let totalVectors = 0;
        Object.values(data.index_metrics).forEach(ns => {
            totalVectors += ns.indexed_vectors || 0;
        });
        html += `
            <div class="config-item-detail">
                <div class="config-label">Total Vectors</div>
                <div class="config-value">${totalVectors.toLocaleString()}</div>
            </div>
        `;
    }

    grid.innerHTML = html || '<div style="grid-column: 1/-1; color: #999;">No configuration data available</div>';
}

// Search API call
async function performSearch() {
    const config = getConfig();
    const query = document.getElementById('searchQuery').value;

    if (!query.trim()) {
        showError('searchResults', 'Please enter a search query');
        return;
    }

    showLoading('searchLoading', true);
    document.getElementById('searchResults').innerHTML = '';

    try {
        const response = await fetch(`${config.apiBase}/search?q=${encodeURIComponent(query)}&k=${config.k}`, {
            headers: { 'x-api-token': config.apiToken }
        });

        if (!response.ok) {
            showError('searchResults', `API Error: ${response.status} ${response.statusText}`);
            return;
        }

        const data = await response.json();
        renderSearchResults(data);
    } catch (err) {
        showError('searchResults', `Error: ${err.message}`, true);
    } finally {
        showLoading('searchLoading', false);
    }
}

// Chat API call
async function performChat() {
    const config = getConfig();
    const question = document.getElementById('chatQuestion').value;

    if (!question.trim()) {
        showError('chatResults', 'Please enter a question');
        return;
    }

    showLoading('chatLoading', true);
    document.getElementById('chatResults').innerHTML = '';

    try {
        const response = await fetch(`${config.apiBase}/chat`, {
            method: 'POST',
            headers: {
                'x-api-token': config.apiToken,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                k: config.k
            })
        });

        if (!response.ok) {
            showError('chatResults', `API Error: ${response.status} ${response.statusText}`);
            return;
        }

        const data = await response.json();
        renderChatResults(data);
    } catch (err) {
        showError('chatResults', `Error: ${err.message}`, true);
    } finally {
        showLoading('chatLoading', false);
    }
}

// Render search results
function renderSearchResults(data) {
    let html = '';

    if (data.results && data.results.length > 0) {
        html += `<div style="margin-bottom: 15px; color: #666; font-size: 0.9em;">
            Found <strong>${data.count}</strong> results (request: <code>${data.request_id}</code>)
        </div>`;

        data.results.forEach(result => {
            html += `
                <div class="result-card">
                    <div style="display: flex; align-items: flex-start;">
                        <span class="result-rank">${result.rank}</span>
                        <div style="flex: 1;">
                            <div class="result-title">${escapeHtml(result.title)}</div>
                            <a href="${result.url}" target="_blank" class="result-url">${result.url}</a>
                            <div class="result-score">Score: ${(result.score * 100).toFixed(1)}%</div>
                        </div>
                    </div>
                </div>
            `;
        });
    } else {
        html += '<div class="error">No results found</div>';
    }

    document.getElementById('searchResults').innerHTML = html;
}

// Render chat results
function renderChatResults(data) {
    let html = '';

    // Answer with citations
    html += `<div class="answer-section">
        <strong>Answer:</strong><br>
        ${escapeHtml(data.answer).replace(/\[(\d+)\]/g, '<sup style="color:#0066cc; font-weight:bold;">[$1]</sup>')}
    </div>`;

    // Sources
    if (data.sources && data.sources.length > 0) {
        html += '<div class="sources-list"><strong>Sources:</strong>';
        data.sources.forEach((source, idx) => {
            const num = idx + 1;
            html += `
                <div class="source-item">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="background: #0066cc; color: white; width: 24px; height: 24px; text-align: center; line-height: 24px; border-radius: 50%; font-weight: bold; font-size: 0.9em;">${num}</span>
                        <div style="flex: 1;">
                            <div class="source-title">${escapeHtml(source.title)}</div>
                            <a href="${source.url}" target="_blank" class="source-url">${source.url}</a>
                        </div>
                    </div>
                </div>
            `;
        });
        html += '</div>';
    }

    // Metadata
    if (data.meta) {
        html += '<div class="meta-info">';
        html += '<strong>Metadata:</strong>';
        html += `<div class="meta-item"><span>Request ID:</span> <code>${data.meta.request_id}</code></div>`;
        html += `<div class="meta-item"><span>Model:</span> ${data.meta.model}</div>`;
        html += `<div class="meta-item"><span>Temperature:</span> ${data.meta.temperature}</div>`;
        if (data.latency_ms) {
            html += `<div class="meta-item"><span>Latency (total):</span> ${data.latency_ms.total}ms (retrieval: ${data.latency_ms.retrieval}ms, LLM: ${data.latency_ms.llm}ms)</div>`;
        }
        html += `<div class="meta-item"><span>Citations found:</span> ${data.citations_found}</div>`;
        html += '</div>';
    }

    document.getElementById('chatResults').innerHTML = html;
}

// Helper functions
function showLoading(elementId, show) {
    document.getElementById(elementId).classList.toggle('active', show);
}

function showError(elementId, message, critical = false) {
    const errorClass = critical ? 'error critical' : 'error';
    document.getElementById(elementId).innerHTML = `<div class="${errorClass}">${escapeHtml(message)}</div>`;
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Enter key support
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('searchQuery').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') performSearch();
    });
    document.getElementById('chatQuestion').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) performChat();
    });
});
