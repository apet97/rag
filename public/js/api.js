/**
 * API Client for Clockify RAG Backend
 */

class RAGApi {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || window.location.origin;
    }

    async search(query, namespace = 'clockify', k = 5) {
        try {
            const params = new URLSearchParams({
                q: query,
                namespace: namespace,
                k: k
            });
            const response = await fetch(`${this.baseUrl}/search?${params}`, {
                headers: { 'x-api-token': 'change-me' }
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Search error:', error);
            throw error;
        }
    }

    async chat(question, namespace = 'clockify', k = 5) {
        try {
            const response = await fetch(`${this.baseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'x-api-token': 'change-me'
                },
                body: JSON.stringify({
                    question: question,
                    namespace: namespace,
                    k: k,
                    allow_rewrites: true,
                    allow_rerank: true
                })
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Chat error:', error);
            throw error;
        }
    }

    async health() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Health check error:', error);
            throw error;
        }
    }
}

const api = new RAGApi();
