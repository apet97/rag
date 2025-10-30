/**
 * Main UI Controller
 */

document.addEventListener('DOMContentLoaded', function() {
    initTabs();
    loadHealth();
});

function initTabs() {
    const tabs = document.querySelectorAll('.tab-btn');
    const panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.dataset.tab;

            // Update buttons
            tabs.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update panels
            panels.forEach(p => p.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
        });
    });
}

async function loadHealth() {
    try {
        const health = await api.health();
        const statusBox = document.getElementById('statusInfo');
        if (statusBox) {
            statusBox.innerHTML = `
                <strong>Status:</strong> ${health.status}<br>
                <strong>Indexes:</strong> ${health.indexes_loaded || 'Loading...'}<br>
                <strong>Last Update:</strong> ${health.last_crawl ? new Date(health.last_crawl).toLocaleString() : 'Never'}
            `;
        }
    } catch (error) {
        console.error('Failed to load health:', error);
        const statusBox = document.getElementById('statusInfo');
        if (statusBox) {
            statusBox.innerHTML = '<p style="color: red;">Unable to connect to server</p>';
        }
    }
}
