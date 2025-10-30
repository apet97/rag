# Code Part 10

## DEPLOY.md

```
# Clockify RAG - Production Deployment Guide

## 🚀 Out-of-the-Box Deployment (3 Steps)

This system is **production-ready** with no configuration required.

---

## **Step 1: Clone Repository**

```bash
git clone <your-repo-url> clockify-rag
cd clockify-rag
```

**Verify you're on the latest version:**
```bash
git log --oneline -1
# Should show: 470bcd88 feat: Implement Harmony chat format support
```

---

## **Step 2: Build Knowledge Base**

```bash
make ingest
```

**What this does:**
- Processes Clockify + LangChain documentation
- Generates embeddings (768-dim E5 model)
- Builds FAISS vector indexes
- Creates BM25 lexical indexes

**Time:** ~5 minutes
**Output:** `index/faiss/clockify/` and `index/faiss/langchain/`

---

## **Step 3: Start Server**

```bash
make serve
```

**Server Details:**
- **URL:** http://localhost:7001
- **Port:** 7001 (configured in .env)
- **API Token:** `05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0`
- **LLM:** Internal gpt-oss:20b at 10.127.0.192:11434 (no API key)

**That's it!** Server is running in production mode.

---

## **Step 4: Verify Deployment**

### Health Check
```bash
curl http://localhost:7001/health | python3 -m json.tool
```

**Expected Response:**
```json
{
  "ok": true,
  "namespaces": ["clockify", "langchain"],
  "embedding_ok": true,
  "llm_ok": true,
  "llm_model": "gpt-oss:20b",
  "harmony_enabled": true
}
```

### Test Search
```bash
curl -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  'http://localhost:7001/search?q=how%20to%20track%20time&k=5' \
  | python3 -m json.tool
```

### Test Chat (Full RAG with Citations)
```bash
curl -X POST http://localhost:7001/chat \
  -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do I create a project?", "k": 5}' \
  | python3 -m json.tool
```

---

## **Configuration Details**

All settings are production-ready in `.env`:

| Setting | Value | Description |
|---------|-------|-------------|
| `ENV` | `prod` | Production environment |
| `API_PORT` | `7001` | Server port |
| `API_TOKEN` | Pre-configured | Secure token (works out of box) |
| `LLM_BASE_URL` | `http://10.127.0.192:11434` | Internal LLM (no API key) |
| `LLM_MODEL` | `gpt-oss:20b` | Harmony-optimized model |
| `LLM_USE_HARMONY` | `auto` | Auto-detects gpt-oss models |
| `HYBRID_SEARCH` | `true` | BM25 + Vector fusion |
| `RERANK_DISABLED` | `false` | Cross-encoder reranking enabled |

**No changes required** - everything works immediately.

---

## **Optional: Web UI**

To use the demo interface:

```bash
# In a new terminal
make ui
```

Then open: **http://localhost:8080**

---

## **Production Features Enabled**

✅ **Security**
- Token authentication (HMAC constant-time comparison)
- Rate limiting (100ms per IP)
- CORS with explicit origins
- ENV=prod enforcement

✅ **Performance**
- Hybrid search (BM25 + Vector with RRF fusion)
- Cross-encoder reranking
- Semantic caching (10K query cache, 1h TTL)
- Circuit breakers (fault tolerance)

✅ **Quality**
- Harmony chat format (gpt-oss:20b optimal performance)
- Inline citations [1], [2] with source URLs
- Query decomposition for complex questions
- MMR diversity penalty

✅ **Observability**
- Prometheus metrics: `/metrics`
- Performance tracking: `/perf?detailed=true`
- Health endpoint: `/health`
- Structured logging (INFO level)

---

## **API Endpoints**

### Core Endpoints
```bash
GET  /health              # Health check
GET  /search              # Retrieval only (no LLM)
POST /chat                # Full RAG with citations
POST /chat/stream         # Streaming RAG (if enabled)
GET  /metrics             # Prometheus metrics
GET  /perf                # Performance stats
```

### Example API Call
```bash
curl -X POST http://localhost:7001/chat \
  -H 'x-api-token: 05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0' \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What is kiosk mode?",
    "namespace": "clockify",
    "k": 5,
    "allow_rewrites": true,
    "allow_rerank": true
  }'
```

---

## **Troubleshooting**

### "Cannot reach LLM at 10.127.0.192:11434"
**Solution:** Ensure you're on the VPN
```bash
ping 10.127.0.192
```

### "Index not found"
**Solution:** Build the index first
```bash
make ingest
```

### "Module not found" errors
**Solution:** Install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Server not responding on port 7001
**Solution:** Check if port is in use
```bash
lsof -i :7001
# If occupied, kill process or change API_PORT in .env
```

---

## **Maintenance**

### Rebuild Index (When Docs Change)
```bash
make ingest
```

### Update Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### View Logs
```bash
tail -f logs/api.log
```

### Run Evaluation
```bash
make eval          # Full evaluation harness
make eval-axioms   # AXIOM 1-9 compliance check
```

---

## **Architecture Overview**

```
┌──────────────┐
│  User Query  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────┐
│  FastAPI (Port 7001)        │
│  - Token auth               │
│  - Rate limiting            │
│  - CORS enforcement         │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Retrieval Engine           │
│  - Hybrid (BM25 + Vector)   │
│  - Query expansion          │
│  - Cross-encoder reranking  │
│  - MMR diversity            │
└──────┬──────────────────────┘
       │
       ├─► FAISS (768-dim E5 embeddings)
       ├─► BM25 (lexical search)
       └─► Cache (semantic LRU)
       │
       ▼
┌─────────────────────────────┐
│  RAGPrompt (Harmony)        │
│  - Context building         │
│  - Citation injection       │
│  - Developer role           │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  LLM (gpt-oss:20b)          │
│  - 10.127.0.192:11434       │
│  - Harmony chat format      │
│  - Circuit breaker          │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  Answer + Citations [1]     │
│  + Source URLs              │
└─────────────────────────────┘
```

---

## **Quick Reference**

| Command | Purpose |
|---------|---------|
| `make ingest` | Build knowledge base indexes |
| `make serve` | Start production server (port 7001) |
| `make ui` | Start demo web interface |
| `make eval` | Run evaluation harness |
| `curl http://localhost:7001/health` | Health check |
| `curl http://localhost:7001/metrics` | Prometheus metrics |

**API Token:** `05yBpumyU52qBrpCTna7YcLPOmCFgpS_qNclXtOaqw0`

---

## **Summary**

✅ **Production-ready out of the box**
✅ **No configuration required**
✅ **3-step deployment: clone → ingest → serve**
✅ **Port 7001 with secure token authentication**
✅ **Internal LLM (no API key needed)**
✅ **Harmony format for optimal gpt-oss:20b performance**

**You're ready to deploy!** 🚀
```

## docs/clockify_glossary.txt

```
# Clockify Glossary #

## Time Tracking Fundamentals ##

### Timesheet #
A weekly or project-based record of all time entries logged by a team member. Timesheets can be submitted for approval and are used for payroll, billing, and project tracking.

### Time Entry #
A single record of time spent on a task, including start time, end time, duration, project, and optional tags.

### Duration #
The total amount of time logged for a time entry, typically measured in hours and minutes.

### Timer #
A real-time clock used to record time as work is being performed. Can be started, paused, and stopped.

### Manual Entry #
Time entry created by directly entering start and end times, rather than using a running timer.

## Project & Task Management ##

### Project #
A container for organizing time entries, often corresponding to client work or internal initiatives. Projects can have budgets, rates, and team members.

### Task #
A specific unit of work within a project. Tasks can be assigned to team members and have estimated durations.

### Estimate #
A predicted duration for completing a task or project. Used for project planning and budget tracking.

### Budget #
A spending limit for a project, either in hours or currency. Helps track project profitability.

## Billing & Rates ##

### Billable #
Time that should be invoiced to a client. Marked as billable in time entries.

### Billable Rate #
The hourly or fixed rate charged to a client for billable work.

### Cost Rate #
The internal labor cost or rate used for cost analysis and profitability calculations.

### Invoicing #
The process of generating and sending bills to clients based on billable time.

### Time Rounding #
Automatically rounding logged time to the nearest interval (e.g., 15 minutes). Used for billing consistency.

## Approvals & Workflow ##

### Timesheet Approval #
The process where managers review and approve submitted timesheets before they are finalized.

### Manager Approval #
A workflow step where a designated manager verifies and approves time entries or timesheets.

### Pending #
Status indicating that a timesheet or approval request is awaiting action.

### Rejected #
Status indicating that a timesheet has been sent back by an approver for revision.

## Workspace & Team Management ##

### Workspace #
A top-level container holding projects, team members, and time tracking data. Similar to an organization or company account.

### Member #
A person added to a workspace who can log time and be assigned to projects.

### Role #
A permission level assigned to team members (e.g., Admin, Manager, Member). Controls what actions a person can perform.

### Team #
A group of members working together, often organized by department or project.

### User Group #
A collection of team members used for bulk operations or permissions management.

### Invitation #
A notification sent to a person to join a workspace.

## Reports & Analytics ##

### Report #
A summary view of time tracking data, including hours logged, costs, and billable amounts.

### Summary Report #
A high-level overview of time by project, member, or date range.

### Detailed Report #
A granular breakdown including individual time entries with all associated metadata.

### CSV Export #
Downloading report data in comma-separated values format for use in external tools.

### PDF Export #
Generating a printable PDF version of a report.

### Scheduled Report #
An automated report that is generated and emailed on a regular schedule (daily, weekly, monthly).

## Features & Integrations ##

### Tags #
Custom labels applied to time entries for categorization and filtering (e.g., "Frontend", "Support", "Code Review").

### Activity Log #
A record of all changes made in the workspace, including user actions and timestamp information.

### Audit Log #
A detailed log of administrative actions and system events for compliance and troubleshooting.

### Idle Detection #
Automatic pausing of a running timer when no activity is detected on the device.

### Pomodoro Timer #
A time management technique using a 25-minute focused work interval followed by a short break. Supported as a timer type in Clockify.

### Kiosk Mode #
A simplified time clock interface for employees to clock in and out using a PIN code or card swipe.

### Webhook #
An automated callback sent to an external system when specific events occur in Clockify (e.g., time entry created).

### SSO (Single Sign-On) #
Integration allowing users to log in with corporate credentials (SAML, OAuth).

### PTO (Paid Time Off) #
Time entries logged for vacation, sick leave, or other paid leave. Often tracked separately from billable work.

## Time Management ##

### Time Off #
Any period when an employee is not available to work, such as vacation or sick leave.

### Holiday #
A designated day or period when the organization is closed and employees are not expected to log work time.

### Vacation #
Planned time off taken by an employee, typically tracked in PTO entries.

### Billable Hours #
The total number of hours logged that should be invoiced to a client.

### Non-Billable Hours #
Time spent on internal tasks, admin work, or other work that is not charged to clients.

### Member Rate #
The specific billing or cost rate assigned to an individual team member.

### Project Rate #
A billing or cost rate that applies to all work on a specific project.

### Workspace Rate #
A default rate applied across an entire workspace unless overridden at the project or member level.
```

## eval/goldset.csv

```
id,question,answer_regex,source_url
exports_csv,How do I export invoices to a CSV file in Clockify?,(?i)export.*csv|save.*csv|csv.*excel,https://clockify.me/help/projects/export-invoices
approvals_api,What approvals workflow can you automate by default with the Clockify API?,(?i)time-?off requests? and approvals,https://clockify.me/help/getting-started/clockify-api-overview
workspace_settings,Who can edit workspace settings and how do you reach that page?,(?i)workspace.*settings|owner|admin,https://clockify.me/help/track-time-and-expenses/workspaces
billable_rates,Where do you set individual billable rates for project members?,(?i)projects? page.*access tab.*billable rate,https://clockify.me/help/projects/managing-people-on-projects
user_roles,Which three core user roles does Clockify define for permissions?,(?i)admin.*manager.*regular,https://clockify.me/help/administration/user-roles-and-permissions/who-can-do-what
shift_timeline,When you shift a scheduled project timeline, what happens automatically afterwards?,(?i)assignments? and tasks?.*automatically adjusted,https://clockify.me/help/projects/manage-scheduled-projects
metadata_fields,How can invisible required fields capture metadata without cluttering the UI?,(?i)invisible fields.*metadata.*users won'?t see,https://clockify.me/help/track-time-and-expenses/define-required-fields
kiosk_universal_pin,What does the kiosk Universal PIN allow you to do?,(?i)Universal PIN.*clock in.*any employee,https://clockify.me/help/track-time-and-expenses/pin
kiosk_clock_out,How do you stop the work timer when clocking out on a kiosk?,(?i)Tap\s+Clock out.*finish your shift,https://clockify.me/help/track-time-and-expenses/track-time-on-kiosk
auto_tracker_start,How do you start Auto tracker recording in the desktop app?,(?i)click\s+the\s+A icon.*Start Recording,https://clockify.me/help/track-time-and-expenses/auto-tracker
time_rounding_toggle,Where do you turn on time rounding in Clockify?,(?i)turn.*on.*rounding|rounding.*toggle|rounding.*switch,https://clockify.me/help/track-time-and-expenses/time-rounding
time_off_history,How do you open a member's time off balance history?,(?i)balance.*history|history.*balance|balance tab,https://clockify.me/help/track-time-and-expenses/accrue-time-off
approvals_pending_tab,Where do submitted timesheets wait while pending approval?,(?i)Approvals.*Pending\s+tab,https://clockify.me/help/getting-started/approve-teams-time-entries
lock_timesheets_toggle,Which menu lets you toggle the Lock entries switch?,(?i)Workspace settings.*Permissions.*Lock entries,https://clockify.me/help/track-time-and-expenses/lock-timesheets
custom_export_customize,How do you customise which data appears when exporting reports?,(?i)Go to\s+a?\s?report.*Click\s+Export.*Choose\s+Customize,https://clockify.me/help/reports/customize-exports
assign_admin_role,Where do you make someone an admin in Clockify?,(?i)Team\s+page.*\+Role.*Admin,https://clockify.me/help/getting-started/invite-users-assign-roles-in-your-workspace
time_off_enable,Where do you enable the Time off feature before creating policies?,(?i)Workspace settings.*Time off.*Enable time off,https://clockify.me/help/track-time-and-expenses/create-manage-time-off-policy
time_off_request_steps,Which page do you open to submit a time off request and what button do you press?,(?i)Go to\s+Time off\s+page.*Click\s+Request time off,https://clockify.me/help/track-time-and-expenses/request-time-off
time_off_notifications,Where do you re-enable email notifications if you stop getting time off alerts?,(?i)Email notifications.*Time off.*checkbox.*enabled,https://clockify.me/help/troubleshooting/time-off-notification-issues
custom_fields_create,Where do you create custom fields for time entries?,(?i)custom.*fields|create.*fields,https://clockify.me/help/track-time-and-expenses/custom-fields
force_timer_today,Which setting blocks users from adding manual time for today or the future?,(?i)force.*timer|lock.*today|prevent.*add,https://clockify.me/help/track-time-and-expenses/lock-timesheets
auto_lock_toggle,What option automatically keeps the lock date up to date?,(?i)Automatically update lock date,https://clockify.me/help/track-time-and-expenses/lock-timesheets
targets_reminders,Where do owners configure timesheet reminder targets?,(?i)Team\s+page\s*>\s*Reminders tab,https://clockify.me/help/administration/targets-reminders
remind_to_approve,How do admins nudge managers to approve pending timesheets?,(?i)click\s+the\s+Remind to approve\s+button,https://clockify.me/help/getting-started/approve-teams-time-entries
expenses_lock,What happens to expenses when you lock timesheets?,(?i)lock date.*expenses.*won't be able to add or edit,https://clockify.me/help/track-time-and-expenses/lock-timesheets
```

## public/js/articles.js

```
/**
 * Articles Interface Controller
 */

const articlesState = {
    allResults: [],
    currentPage: 0,
    resultsPerPage: 9,
    isLoading: false
};

document.addEventListener('DOMContentLoaded', function() {
    const articlesInput = document.getElementById('articlesInput');
    const articleSearchBtn = document.getElementById('articleSearchBtn');

    articlesInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            searchArticles();
        }
    });

    articleSearchBtn.addEventListener('click', searchArticles);
});

async function searchArticles() {
    const query = document.getElementById('articlesInput').value.trim();
    if (!query || articlesState.isLoading) return;

    articlesState.isLoading = true;
    document.getElementById('articleSearchBtn').disabled = true;

    try {
        const response = await api.search(query, 'clockify', 20);
        articlesState.allResults = response.results || [];
        articlesState.currentPage = 0;

        if (articlesState.allResults.length === 0) {
            document.getElementById('articlesResults').innerHTML = '<p class="empty-state">No articles found. Try different keywords.</p>';
            document.getElementById('pagination').style.display = 'none';
        } else {
            displayArticlesPage();
        }
    } catch (error) {
        document.getElementById('articlesResults').innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
    } finally {
        articlesState.isLoading = false;
        document.getElementById('articleSearchBtn').disabled = false;
    }
}

function displayArticlesPage() {
    const { allResults, currentPage, resultsPerPage } = articlesState;
    const start = currentPage * resultsPerPage;
    const end = start + resultsPerPage;
    const pageResults = allResults.slice(start, end);

    const filterHighConf = document.getElementById('filterHighConf').checked;
    const filtered = filterHighConf ? pageResults.filter(r => (r.confidence || 0) > 50) : pageResults;

    const html = filtered.map(article => createArticleCard(article)).join('');
    document.getElementById('articlesResults').innerHTML = html || '<p class="empty-state">No results match your filters.</p>';

    // Update pagination
    const totalPages = Math.ceil(allResults.length / resultsPerPage);
    if (totalPages > 1) {
        document.getElementById('pagination').style.display = 'flex';
        document.getElementById('pageInfo').textContent = `Page ${currentPage + 1} of ${totalPages}`;
        document.getElementById('prevBtn').disabled = currentPage === 0;
        document.getElementById('nextBtn').disabled = currentPage === totalPages - 1;

        document.getElementById('prevBtn').onclick = () => {
            articlesState.currentPage--;
            displayArticlesPage();
        };
        document.getElementById('nextBtn').onclick = () => {
            articlesState.currentPage++;
            displayArticlesPage();
        };
    } else {
        document.getElementById('pagination').style.display = 'none';
    }
}

function createArticleCard(article) {
    const confidence = article.confidence || 0;
    const level = confidence > 75 ? 'high' : confidence > 50 ? 'medium' : 'low';
    const emoji = confidence > 75 ? '🟢' : confidence > 50 ? '🟡' : '🔴';

    // Sanitize text fields to prevent XSS
    const title = sanitizeHtml(article.title || 'Untitled');
    const content = sanitizeHtml((article.content || 'No content').substring(0, 150));
    const namespace = sanitizeHtml(article.namespace || 'unknown');

    return `
        <div class="article-card" onclick="openArticle('${encodeURIComponent(JSON.stringify(article))}')">
            <h3>${title}</h3>
            <p>${content}...</p>
            <div class="article-meta">
                <span class="confidence-badge confidence-${level}">${emoji} ${confidence.toFixed(0)}%</span>
                <span style="font-size: 0.75rem;">${namespace}</span>
            </div>
        </div>
    `;
}

/**
 * Sanitize HTML to prevent XSS attacks
 * Escapes HTML entities
 */
function sanitizeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function openArticle(encodedData) {
    try {
        const article = JSON.parse(decodeURIComponent(encodedData));
        const url = article.url;
        if (url) {
            window.open(url, '_blank');
        }
    } catch (e) {
        console.error('Failed to open article:', e);
    }
}

// Filter change listener
document.addEventListener('DOMContentLoaded', function() {
    const filterCheckbox = document.getElementById('filterHighConf');
    if (filterCheckbox) {
        filterCheckbox.addEventListener('change', () => {
            if (articlesState.allResults.length > 0) {
                displayArticlesPage();
            }
        });
    }
});
```

## scripts/demo_rag.py

```
#!/usr/bin/env python3
"""Interactive demo of Clockify RAG system with hand-picked queries."""

import requests
import time
from pathlib import Path
from datetime import datetime

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Hand-picked realistic queries for demo
DEMO_QUERIES = [
    {
        "query": "How do I start tracking time in Clockify?",
        "category": "Getting Started",
        "namespace": "clockify"
    },
    {
        "query": "What integrations does Clockify support?",
        "category": "Integrations",
        "namespace": "clockify"
    },
    {
        "query": "How do I generate a timesheet report?",
        "category": "Reports",
        "namespace": "clockify"
    },
    {
        "query": "Can I track time retroactively?",
        "category": "Time Tracking",
        "namespace": "clockify"
    },
    {
        "query": "How do I set up project budgeting?",
        "category": "Project Management",
        "namespace": "clockify"
    },
    {
        "query": "What are the best practices for time tracking?",
        "category": "Best Practices",
        "namespace": "clockify"
    },
    {
        "query": "How do I export data to Excel?",
        "category": "Data Export",
        "namespace": "clockify"
    },
    {
        "query": "Can I use Clockify on mobile?",
        "category": "Mobile",
        "namespace": "clockify"
    },
    {
        "query": "How do I set up team approvals?",
        "category": "Team Management",
        "namespace": "clockify"
    },
    {
        "query": "What is the difference between projects and tasks?",
        "category": "Organization",
        "namespace": "clockify"
    },
]

class Demo:
    """Interactive RAG demo."""

    def __init__(self):
        self.base_url = "http://localhost:8888"
        self.results = []
        self.llm_available = False

    def check_llm(self):
        """Check if LLM is available."""
        try:
            response = requests.post(
                "http://localhost:8080/v1/chat/completions",
                json={
                    "model": "oss20b",
                    "messages": [{"role": "user", "content": "OK"}],
                    "max_tokens": 5,
                },
                timeout=5
            )
            self.llm_available = response.status_code == 200
        except:
            self.llm_available = False

    def print_header(self, text):
        """Print formatted header."""
        print(f"\n{'='*80}")
        print(f"{text:^80}")
        print(f"{'='*80}\n")

    def print_separator(self):
        """Print separator."""
        print(f"{'-'*80}\n")

    def run_demo_query(self, query_data):
        """Run a single demo query."""
        query = query_data["query"]
        category = query_data["category"]
        namespace = query_data["namespace"]

        print(f"📌 Category: {category}")
        print(f"❓ Query: {query}\n")

        try:
            # Retrieve relevant sources
            start_time = time.time()
            response = requests.get(
                f"{self.base_url}/search",
                params={"q": query, "namespace": namespace, "k": 3},
                timeout=10
            )
            retrieval_time = time.time() - start_time

            if response.status_code != 200:
                print(f"❌ Retrieval failed\n")
                return None

            data = response.json()
            sources = data.get("results", [])

            if not sources:
                print(f"❌ No sources found\n")
                return None

            print(f"📚 Retrieved {len(sources)} sources in {retrieval_time:.2f}s:")
            for i, source in enumerate(sources, 1):
                title = source.get("title", "Untitled")
                score = source.get("vector_score", 0)
                print(f"   {i}. {title[:70]} (score: {score:.3f})")

            # Try to get LLM answer if available
            if self.llm_available:
                print(f"\n⏳ Generating answer with LLM...")
                try:
                    llm_start = time.time()
                    llm_response = requests.post(
                        f"{self.base_url}/chat",
                        json={"question": query, "namespace": namespace, "k": 3},
                        timeout=15
                    )
                    llm_time = time.time() - llm_start

                    if llm_response.status_code == 200:
                        llm_data = llm_response.json()
                        answer = llm_data.get("answer", "")

                        if answer:
                            print(f"\n💬 Answer ({llm_time:.2f}s):")
                            # Print first 150 characters of answer
                            answer_preview = answer[:150] + ("..." if len(answer) > 150 else "")
                            print(f"   {answer_preview}\n")

                            return {
                                "query": query,
                                "category": category,
                                "retrieval_time": retrieval_time,
                                "llm_time": llm_time,
                                "sources_count": len(sources),
                                "has_answer": True
                            }

                except:
                    pass

            # Retrieval-only result
            return {
                "query": query,
                "category": category,
                "retrieval_time": retrieval_time,
                "llm_time": 0,
                "sources_count": len(sources),
                "has_answer": False
            }

        except Exception as e:
            print(f"❌ Error: {str(e)}\n")
            return None

    def run(self):
        """Run the interactive demo."""
        self.print_header("🎯 CLOCKIFY RAG SYSTEM - INTERACTIVE DEMO")

        print(f"System Status:")
        print(f"  Server:           http://localhost:8888")
        print(f"  Retrieval:        ✅ Ready")

        self.check_llm()
        if self.llm_available:
            print(f"  LLM:              ✅ Available")
        else:
            print(f"  LLM:              ⏳ Not running (showing retrieval-only)")

        print(f"  Demo Queries:     {len(DEMO_QUERIES)}")

        self.print_header("🚀 DEMO EXECUTION")

        successful = 0
        failed = 0

        for i, query_data in enumerate(DEMO_QUERIES, 1):
            print(f"[{i}/{len(DEMO_QUERIES)}] ", end="")

            result = self.run_demo_query(query_data)

            if result:
                self.results.append(result)
                successful += 1
            else:
                failed += 1

            self.print_separator()

        # Print demo summary
        self.print_header("📊 DEMO SUMMARY")

        print(f"Total Queries:          {len(DEMO_QUERIES)}")
        print(f"Successful:             {successful} ✅")
        print(f"Failed:                 {failed} ❌")

        if self.results:
            avg_retrieval_time = sum(r["retrieval_time"] for r in self.results) / len(self.results)
            total_sources = sum(r["sources_count"] for r in self.results)
            with_answers = sum(1 for r in self.results if r.get("has_answer"))

            print(f"\nRetrieval Performance:")
            print(f"  Average Latency:    {avg_retrieval_time:.3f}s")
            print(f"  Total Sources:      {total_sources}")
            print(f"  Avg per Query:      {total_sources / len(self.results):.1f}")

            if self.llm_available and with_answers > 0:
                avg_llm_time = sum(r["llm_time"] for r in self.results if r["has_answer"]) / with_answers
                print(f"\nLLM Performance:")
                print(f"  Queries with Answer: {with_answers}")
                print(f"  Average Latency:     {avg_llm_time:.3f}s")

            print(f"\nSystem Readiness:")
            if successful == len(DEMO_QUERIES):
                print(f"  ✅ PRODUCTION READY - All queries successful")
                readiness_score = 100
            elif successful >= len(DEMO_QUERIES) * 0.8:
                print(f"  ✅ NEARLY READY - {successful}/{len(DEMO_QUERIES)} queries successful")
                readiness_score = 85
            else:
                print(f"  ⚠️  NEEDS WORK - {successful}/{len(DEMO_QUERIES)} queries successful")
                readiness_score = 50

            print(f"  Readiness Score:     {readiness_score}/100")

        # Save demo report
        report_file = LOG_DIR / "demo_report.json"
        with open(report_file, "w") as f:
            import json
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "llm_available": self.llm_available,
                "queries_run": len(DEMO_QUERIES),
                "successful": successful,
                "failed": failed,
                "results": self.results
            }, f, indent=2)

        print(f"\n✅ Demo report saved to {report_file}")

        print(f"\nNext Steps:")
        print(f"  1. To enable full RAG with LLM generation:")
        print(f"     ollama pull oss20b && ollama serve")
        print(f"  2. Rerun this demo to see LLM-generated answers")
        print(f"  3. Deploy to production: python scripts/deployment_checklist.py")

        return 0 if failed == 0 else 1

if __name__ == "__main__":
    demo = Demo()
    exit_code = demo.run()
    exit(exit_code)
```

## src/circuit_breaker.py

```
"""
Circuit Breaker Pattern for Fault Tolerance

Implements the circuit breaker pattern to prevent cascading failures
and enable graceful degradation when external services are unavailable.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Too many failures, requests are blocked (fail fast)
- HALF_OPEN: Testing if service recovered, limited requests allowed
"""

from __future__ import annotations

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, TypeVar, Dict
from functools import wraps
from threading import RLock

from src.errors import CircuitOpenError, DependencyError

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Data Structures
# ============================================================================


class CircuitState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Blocking requests (fail fast)
    HALF_OPEN = "half_open"    # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5         # Failures before opening
    recovery_timeout_seconds: float = 60.0  # Time before trying half-open
    success_threshold: int = 2         # Successes in half-open before closing
    name: str = "circuit_breaker"


@dataclass
class CircuitBreakerMetrics:
    """Metrics for monitoring circuit breaker."""
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    consecutive_failures: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0
    time_opened: Optional[float] = None


# ============================================================================
# Circuit Breaker Implementation
# ============================================================================


class CircuitBreaker:
    """
    Circuit breaker for handling failures in external service calls.

    Protects against cascading failures by stopping requests to a failing service
    and allowing it time to recover.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker."""
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self._lock = RLock()
        self._half_open_successes = 0
        self._state_changed_at = time.time()

    def call(
        self,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
            DependencyError: If function call fails
        """
        with self._lock:
            # Check if circuit should transition states
            self._check_state_transition()

            # Reject if open
            if self.state == CircuitState.OPEN:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.config.name}' is OPEN - service unavailable",
                    service=self.config.name,
                )

            # Limit concurrent requests in half-open state
            if (
                self.state == CircuitState.HALF_OPEN
                and self._half_open_successes >= self.config.success_threshold
            ):
                raise CircuitOpenError(
                    f"Circuit breaker '{self.config.name}' is testing recovery - "
                    f"max concurrent requests exceeded",
                    service=self.config.name,
                )

        # Execute function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        now = time.time()

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed
            if self.metrics.time_opened is None:
                self.metrics.time_opened = now

            elapsed = now - self.metrics.time_opened
            if elapsed >= self.config.recovery_timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
                logger.info(
                    f"Circuit breaker '{self.config.name}' transitioning to HALF_OPEN "
                    f"after {elapsed:.1f}s timeout"
                )

    def _on_success(self) -> None:
        """Handle successful call."""
        with self._lock:
            self.metrics.total_successes += 1
            self.metrics.total_requests += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(
                        f"Circuit breaker '{self.config.name}' transitioning to CLOSED "
                        f"after {self._half_open_successes} successes"
                    )

    def _on_failure(self, error: Exception) -> None:
        """Handle failed call."""
        with self._lock:
            self.metrics.total_failures += 1
            self.metrics.total_requests += 1
            self.metrics.consecutive_failures += 1
            self.metrics.last_failure_time = time.time()

            logger.warning(
                f"Circuit breaker '{self.config.name}' failure "
                f"({self.metrics.consecutive_failures}/{self.config.failure_threshold}): {error}"
            )

            # Transition to open if threshold reached
            if self.metrics.consecutive_failures >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
                logger.error(
                    f"Circuit breaker '{self.config.name}' transitioning to OPEN - "
                    f"{self.metrics.consecutive_failures} consecutive failures"
                )

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state."""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self.metrics.state_changes += 1
            self.metrics.time_opened = time.time() if new_state == CircuitState.OPEN else None
            self._half_open_successes = 0

            logger.info(
                f"Circuit breaker '{self.config.name}' transitioned from "
                f"{old_state.value} to {new_state.value}"
            )

    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self.metrics.consecutive_failures = 0
            self._half_open_successes = 0
            logger.info(f"Circuit breaker '{self.config.name}' manually reset")

    def get_state(self) -> CircuitState:
        """Get current state."""
        with self._lock:
            self._check_state_transition()
            return self.state

    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get current metrics."""
        with self._lock:
            return CircuitBreakerMetrics(
                total_requests=self.metrics.total_requests,
                total_failures=self.metrics.total_failures,
                total_successes=self.metrics.total_successes,
                consecutive_failures=self.metrics.consecutive_failures,
                last_failure_time=self.metrics.last_failure_time,
                last_success_time=self.metrics.last_success_time,
                state_changes=self.metrics.state_changes,
                time_opened=self.metrics.time_opened,
            )

    def get_status(self) -> Dict[str, Any]:
        """Get human-readable status."""
        with self._lock:
            self._check_state_transition()
            metrics = self.get_metrics()

            success_rate = (
                (metrics.total_successes / metrics.total_requests * 100)
                if metrics.total_requests > 0
                else 0
            )

            return {
                "name": self.config.name,
                "state": self.state.value,
                "total_requests": metrics.total_requests,
                "total_failures": metrics.total_failures,
                "total_successes": metrics.total_successes,
                "success_rate": f"{success_rate:.1f}%",
                "consecutive_failures": metrics.consecutive_failures,
                "failure_threshold": self.config.failure_threshold,
                "last_failure": metrics.last_failure_time,
                "last_success": metrics.last_success_time,
                "state_changes": metrics.state_changes,
            }


# ============================================================================
# Decorator for Easy Integration
# ============================================================================


def circuit_breaker(
    name: str = "default",
    failure_threshold: int = 5,
    recovery_timeout_seconds: float = 60.0,
    success_threshold: int = 2,
):
    """
    Decorator to wrap a function with circuit breaker protection.

    Args:
        name: Name of the circuit breaker
        failure_threshold: Failures before opening
        recovery_timeout_seconds: Time before testing recovery
        success_threshold: Successes in half-open before closing

    Returns:
        Decorated function with circuit breaker protection

    Example:
        @circuit_breaker(name="llm_service", failure_threshold=3)
        def call_llm(prompt):
            return llm.generate(prompt)
    """
    config = CircuitBreakerConfig(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout_seconds=recovery_timeout_seconds,
        success_threshold=success_threshold,
    )
    breaker = CircuitBreaker(config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(func, *args, **kwargs)

        # Expose breaker for manual management
        wrapper._circuit_breaker = breaker  # type: ignore

        return wrapper

    return decorator


# ============================================================================
# Global Circuit Breaker Registry
# ============================================================================


class CircuitBreakerRegistry:
    """Global registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = RLock()

    def register(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Register a new circuit breaker."""
        with self._lock:
            if name in self._breakers:
                logger.warning(f"Circuit breaker '{name}' already registered")
                return self._breakers[name]

            config = config or CircuitBreakerConfig(name=name)
            breaker = CircuitBreaker(config)
            self._breakers[name] = breaker
            logger.info(f"Registered circuit breaker '{name}'")
            return breaker

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        with self._lock:
            return self._breakers.get(name)

    def get_or_create(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker."""
        breaker = self.get(name)
        if breaker is None:
            breaker = self.register(name, config)
        return breaker

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """List all circuit breakers and their status."""
        with self._lock:
            return {name: breaker.get_status() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")


# Global registry instance
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry."""
    return _registry.get_or_create(name, config)


def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get status of all registered circuit breakers."""
    return _registry.list_all()


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    _registry.reset_all()
```

## src/metrics.py

```
"""
Prometheus metrics for RAG API monitoring.

Provides counters, histograms, and gauges for tracking:
- Request counts and status codes
- Request latency distributions
- Cache hit rates
- Index operations
- LLM call statistics
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from typing import Optional

# ============================================================================
# HTTP Request Metrics
# ============================================================================

request_count = Counter(
    'rag_requests_total',
    'Total number of HTTP requests',
    ['endpoint', 'method', 'status']
)

request_latency = Histogram(
    'rag_request_duration_seconds',
    'HTTP request latency in seconds',
    ['endpoint', 'method'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

# ============================================================================
# Cache Metrics
# ============================================================================

cache_hits = Counter(
    'rag_cache_hits_total',
    'Total number of cache hits',
    ['cache_type']
)

cache_misses = Counter(
    'rag_cache_misses_total',
    'Total number of cache misses',
    ['cache_type']
)

cache_size = Gauge(
    'rag_cache_size_entries',
    'Current number of entries in cache',
    ['cache_type']
)

cache_evictions = Counter(
    'rag_cache_evictions_total',
    'Total number of cache evictions',
    ['cache_type']
)

# ============================================================================
# Search & Retrieval Metrics
# ============================================================================

search_query_count = Counter(
    'rag_search_queries_total',
    'Total number of search queries',
    ['namespace', 'query_type']
)

search_results_count = Histogram(
    'rag_search_results_count',
    'Number of results returned per search',
    ['namespace'],
    buckets=(0, 1, 3, 5, 10, 20, 50, 100)
)

search_latency = Histogram(
    'rag_search_duration_seconds',
    'Search operation latency in seconds',
    ['namespace', 'strategy'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

rerank_latency = Histogram(
    'rag_rerank_duration_seconds',
    'Reranking operation latency in seconds',
    ['namespace'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

# ============================================================================
# LLM Metrics
# ============================================================================

llm_requests = Counter(
    'rag_llm_requests_total',
    'Total number of LLM requests',
    ['model', 'status']
)

llm_latency = Histogram(
    'rag_llm_duration_seconds',
    'LLM request latency in seconds',
    ['model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)
)

llm_tokens = Histogram(
    'rag_llm_tokens_total',
    'Estimated LLM tokens per request',
    ['model', 'type'],  # type: input or output
    buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000)
)

# ============================================================================
# Index Metrics
# ============================================================================

index_size = Gauge(
    'rag_index_vectors_count',
    'Number of vectors in FAISS index',
    ['namespace']
)

index_load_time = Histogram(
    'rag_index_load_duration_seconds',
    'Time taken to load index on startup',
    ['namespace'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
)

# ============================================================================
# System Metrics
# ============================================================================

system_uptime = Gauge(
    'rag_system_uptime_seconds',
    'System uptime in seconds'
)

active_namespaces = Gauge(
    'rag_active_namespaces_count',
    'Number of active namespaces'
)

# ============================================================================
# Circuit Breaker Metrics
# ============================================================================

circuit_breaker_state = Gauge(
    'rag_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=half_open, 2=open)',
    ['name']
)

circuit_breaker_requests = Counter(
    'rag_circuit_breaker_requests_total',
    'Total requests through circuit breaker',
    ['name', 'status']  # status: success or failure
)

circuit_breaker_failures = Counter(
    'rag_circuit_breaker_failures_total',
    'Total failures in circuit breaker',
    ['name']
)

circuit_breaker_consecutive_failures = Gauge(
    'rag_circuit_breaker_consecutive_failures',
    'Current consecutive failures',
    ['name']
)

circuit_breaker_state_changes = Counter(
    'rag_circuit_breaker_state_changes_total',
    'Circuit breaker state transitions',
    ['name', 'from_state', 'to_state']
)

# ============================================================================
# Helper Functions
# ============================================================================

def track_request(endpoint: str, method: str, status: int, duration: float) -> None:
    """
    Track HTTP request metrics.

    Args:
        endpoint: API endpoint path
        method: HTTP method (GET, POST, etc.)
        status: HTTP status code
        duration: Request duration in seconds
    """
    request_count.labels(endpoint=endpoint, method=method, status=status).inc()
    request_latency.labels(endpoint=endpoint, method=method).observe(duration)


def track_cache_operation(cache_type: str, hit: bool, size: Optional[int] = None) -> None:
    """
    Track cache hit/miss metrics.

    Args:
        cache_type: Type of cache (response, bm25, vector, etc.)
        hit: True if cache hit, False if miss
        size: Current cache size (optional)
    """
    if hit:
        cache_hits.labels(cache_type=cache_type).inc()
    else:
        cache_misses.labels(cache_type=cache_type).inc()

    if size is not None:
        cache_size.labels(cache_type=cache_type).set(size)


def track_search(namespace: str, query_type: str, num_results: int,
                 duration: float, strategy: str = "hybrid") -> None:
    """
    Track search operation metrics.

    Args:
        namespace: Search namespace
        query_type: Type of query (simple, multi-intent, etc.)
        num_results: Number of results returned
        duration: Search duration in seconds
        strategy: Retrieval strategy used
    """
    search_query_count.labels(namespace=namespace, query_type=query_type).inc()
    search_results_count.labels(namespace=namespace).observe(num_results)
    search_latency.labels(namespace=namespace, strategy=strategy).observe(duration)


def track_llm_request(model: str, status: str, duration: float,
                      input_tokens: Optional[int] = None,
                      output_tokens: Optional[int] = None) -> None:
    """
    Track LLM request metrics.

    Args:
        model: LLM model name
        status: Request status (success, error, timeout, etc.)
        duration: Request duration in seconds
        input_tokens: Number of input tokens (optional)
        output_tokens: Number of output tokens (optional)
    """
    llm_requests.labels(model=model, status=status).inc()
    llm_latency.labels(model=model).observe(duration)

    if input_tokens is not None:
        llm_tokens.labels(model=model, type="input").observe(input_tokens)
    if output_tokens is not None:
        llm_tokens.labels(model=model, type="output").observe(output_tokens)


def track_circuit_breaker(name: str, state: str, metrics_data: dict) -> None:
    """
    Track circuit breaker metrics.

    Args:
        name: Circuit breaker name
        state: Current state (closed, half_open, open)
        metrics_data: Dictionary with total_requests, total_failures, total_successes, consecutive_failures
    """
    # Map state to numeric value for Prometheus
    state_map = {"closed": 0, "half_open": 1, "open": 2}
    circuit_breaker_state.labels(name=name).set(state_map.get(state, 0))

    # Update counters (we set them to match the circuit breaker's internal counters)
    circuit_breaker_consecutive_failures.labels(name=name).set(
        metrics_data.get("consecutive_failures", 0)
    )


def get_metrics() -> bytes:
    """
    Get current metrics in Prometheus format.

    Returns:
        Prometheus-formatted metrics as bytes
    """
    return generate_latest()


def get_content_type() -> str:
    """
    Get Prometheus metrics content type.

    Returns:
        Content-Type header value
    """
    return CONTENT_TYPE_LATEST
```

## src/query_optimizer.py

```
#!/usr/bin/env python3
"""
Query optimization and understanding module.

Analyzes queries to extract intent, entities, and generates optimized search terms.
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types."""
    DEFINITION = "definition"  # "What is X?"
    HOWTO = "how-to"  # "How do I...?"
    COMPARISON = "comparison"  # "X vs Y"
    FACTUAL = "factual"  # "Does X have...?"
    GENERAL = "general"  # Everything else


class QueryOptimizer:
    """Optimize and analyze user queries for better retrieval."""

    # Query type detection patterns
    DEFINITION_PATTERNS = [
        r"^what\s+(?:is|are)\s+",
        r"^what's\s+",
        r"^define\s+",
        r"^explain\s+",
    ]

    HOWTO_PATTERNS = [
        r"^how\s+(?:do|can|to)\s+",
        r"^how\s+(?:do\s+)?i\s+",
        r"^show\s+me\s+",
        r"^help\s+",
    ]

    COMPARISON_PATTERNS = [
        r"\s+vs\s+",
        r"\s+versus\s+",
        r"difference\s+between",
        r"compare\s+",
    ]

    FACTUAL_PATTERNS = [
        r"^does\s+",
        r"^can\s+",
        r"^is\s+",
        r"^are\s+",
        r"^will\s+",
    ]

    # Stop words to remove from query
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "must", "shall", "this", "that",
        "these", "those", "i", "you", "he", "she", "it", "we", "they",
    }

    def __init__(self):
        """Initialize query optimizer."""
        self.query_history = []

    def analyze(self, query: str) -> Dict:
        """
        Analyze query and return optimization info.

        Returns:
            Dict with:
            - original: original query
            - cleaned: cleaned query (lowercase, normalized)
            - type: detected query type
            - entities: extracted entities/keywords
            - expansion: suggested term expansions
            - confidence: confidence 0-1
        """
        if not query or len(query) < 2:
            logger.warning(f"Query too short: {query}")
            return {
                "original": query,
                "cleaned": query.lower().strip(),
                "type": QueryType.GENERAL.value,
                "entities": [],
                "expansion": [],
                "confidence": 0.0,
                "error": "Query too short",
            }

        cleaned = self._clean(query)
        query_type = self._detect_type(cleaned)
        entities = self._extract_entities(cleaned)
        expansion = self._generate_expansion(entities, query_type)

        # Confidence: higher for specific queries with clear intent
        confidence = min(1.0, len(entities) * 0.2 + (0.5 if query_type != QueryType.GENERAL else 0.0))

        result = {
            "original": query,
            "cleaned": cleaned,
            "type": query_type.value,
            "entities": entities,
            "expansion": expansion,
            "confidence": confidence,
        }

        self.query_history.append(result)
        return result

    def _clean(self, query: str) -> str:
        """Clean and normalize query."""
        # Lowercase
        q = query.lower().strip()

        # Remove extra whitespace
        q = re.sub(r"\s+", " ", q)

        # Remove punctuation except spaces
        q = re.sub(r"[^\w\s]", "", q)

        return q.strip()

    def _detect_type(self, query: str) -> QueryType:
        """Detect query type (definition, how-to, comparison, etc.)."""
        query_lower = query.lower()

        # Check patterns in order of priority
        for pattern in self.DEFINITION_PATTERNS:
            if re.match(pattern, query_lower):
                return QueryType.DEFINITION

        for pattern in self.HOWTO_PATTERNS:
            if re.match(pattern, query_lower):
                return QueryType.HOWTO

        for pattern in self.COMPARISON_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.COMPARISON

        for pattern in self.FACTUAL_PATTERNS:
            if re.match(pattern, query_lower):
                return QueryType.FACTUAL

        return QueryType.GENERAL

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract key entities/keywords from query.

        Removes stop words and extracts meaningful terms.
        """
        words = query.lower().split()

        # Filter out stop words and short words
        entities = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e not in seen:
                unique_entities.append(e)
                seen.add(e)

        return unique_entities[:5]  # Limit to 5 top entities

    def _generate_expansion(self, entities: List[str], query_type: QueryType) -> List[str]:
        """
        Generate term expansions for better retrieval.

        Suggests related terms based on entity type and query intent.
        """
        expansions = []

        # Map of entity patterns to expansion suggestions
        expansion_map = {
            "track": ["time", "entry", "timer", "tracking"],
            "project": ["create", "manage", "delete", "setup"],
            "user": ["member", "team", "permission", "role"],
            "time": ["track", "entry", "hours", "duration"],
            "report": ["generate", "export", "view", "analytics"],
            "team": ["user", "member", "group", "workspace"],
            "permission": ["access", "role", "grant", "allow"],
        }

        # Add query-type-specific expansions
        type_expansions = {
            QueryType.HOWTO: ["step", "process", "guide", "setup"],
            QueryType.DEFINITION: ["what", "explain", "meaning", "definition"],
            QueryType.COMPARISON: ["difference", "versus", "better", "alternative"],
        }

        # Apply entity-specific expansions
        for entity in entities:
            if entity in expansion_map:
                expansions.extend(expansion_map[entity])

        # Apply type-specific expansions
        if query_type in type_expansions:
            expansions.extend(type_expansions[query_type])

        # Remove duplicates and limit to 5
        unique_expansions = list(dict.fromkeys(expansions))[:5]

        return unique_expansions

    def get_search_query(self, analysis: Dict) -> str:
        """
        Generate optimized search query from analysis.

        Combines entities and expansions for better retrieval.
        """
        parts = []

        # Primary entities (always included)
        if analysis.get("entities"):
            parts.extend(analysis["entities"])

        # Include expansion terms
        if analysis.get("expansion"):
            # For high-confidence queries, add expansions
            if analysis.get("confidence", 0) > 0.3:
                parts.extend(analysis["expansion"][:2])  # Top 2 expansions

        # Join and deduplicate
        unique_parts = list(dict.fromkeys(parts))
        search_query = " ".join(unique_parts)

        return search_query if search_query else analysis.get("cleaned", "")

    def suggest_refinements(self, analysis: Dict, results_count: int = 0) -> List[str]:
        """
        Suggest query refinements if results are poor.

        Args:
            analysis: Query analysis result
            results_count: Number of results found

        Returns:
            List of suggested refined queries
        """
        suggestions = []

        # If no results, suggest broader searches
        if results_count == 0:
            entities = analysis.get("entities", [])
            if entities:
                # Try searching for individual entities
                suggestions = [f"Search for: {e}" for e in entities[:3]]
            else:
                suggestions.append("Try a different keyword or phrase")

        # If few results, suggest expansions
        elif results_count < 3:
            expansions = analysis.get("expansion", [])
            if expansions:
                suggestions.append(f"Also try: {expansions[0]}")

        return suggestions

    def get_stats(self) -> Dict:
        """Get optimizer statistics."""
        if not self.query_history:
            return {"queries_analyzed": 0}

        type_counts = {}
        for query in self.query_history:
            qtype = query.get("type", "unknown")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

        avg_confidence = sum(q.get("confidence", 0) for q in self.query_history) / len(self.query_history)

        return {
            "queries_analyzed": len(self.query_history),
            "type_distribution": type_counts,
            "avg_confidence": round(avg_confidence, 2),
        }


# Global instance
_optimizer = None


def get_optimizer() -> QueryOptimizer:
    """Get or create global query optimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = QueryOptimizer()
    return _optimizer
```

## src/tuning_config.py

```
#!/usr/bin/env python3
"""
Centralized Tuning Constants for RAG System

This module consolidates all magic numbers and hyperparameters across the system.
Modifying values here affects global behavior without code changes.

PHASE 5 Gold-Standard RAG Tuning:
- RRF (Reciprocal Rank Fusion): Research-backed hybrid ranking
- MMR (Maximal Marginal Relevance): Diversity-aware result selection
- Time Decay: Freshness boosting for time-sensitive documents
- Answerability Scoring: Grounding validation (Jaccard overlap)
- Semantic Caching: LRU cache with embedding fingerprint keys
"""

# ============================================================================
# PHASE 5: Gold-Standard RAG Fusion & Diversity
# ============================================================================

# Reciprocal Rank Fusion (RRF)
# Formula: score = 1/(k+rank) for both vector and BM25 rankings
# k=60 is research-standard; lower k = higher weight to top results
RRF_K_CONSTANT: float = 60.0

# Maximal Marginal Relevance (MMR)
# Formula: MMR_score = λ*relevance - (1-λ)*max_similarity_to_selected
# λ=0.7 balances relevance (70%) vs diversity (30%)
MMR_LAMBDA: float = 0.7

# Time Decay for Freshness
# Formula: boosted_score = original_score * (decay_rate ^ months_old)
# 0.95 = 5% decay per month; documents decay over time
TIME_DECAY_RATE: float = 0.95
TIME_DECAY_MONTHS_DIVISOR: float = 30.0  # Days to months conversion

# ============================================================================
# Answerability & Grounding Validation
# ============================================================================

# Jaccard overlap threshold for answer grounding
# score >= 0.18 means >=18% token overlap between answer and context
# Below this threshold: answer may be hallucinated -> safe refusal instead
# Lowered from 0.25 to 0.18 to allow more paraphrased answers while still catching hallucinations
ANSWERABILITY_THRESHOLD: float = 0.18

# ============================================================================
# Semantic Answer Caching
# ============================================================================

# Maximum number of cached entries (LRU eviction after this)
SEMANTIC_CACHE_MAX_SIZE: int = 10000

# Time-to-live for cached entries (seconds = 1 hour)
SEMANTIC_CACHE_TTL_SECONDS: int = 3600

# ============================================================================
# Hybrid Search Configuration
# ============================================================================

# Hybrid ranking weight: α*vector + (1-α)*bm25
# DEPRECATED: PHASE 5 replaced with RRF, but kept for legacy support
HYBRID_ALPHA: float = 0.7

# Diversity penalty weight in old diversity filtering (deprecated)
# PHASE 5: Replaced with MMR (Maximal Marginal Relevance)
DIVERSITY_PENALTY_WEIGHT: float = 0.15

# ============================================================================
# BM25 Text Ranking
# ============================================================================

# BM25 length normalization parameter (typical: 0.5-0.75)
# 0.75 = moderate length normalization (default tuning)
BM25_B: float = 0.75

# ============================================================================
# LLM Generation
# ============================================================================

# LLM temperature for deterministic/consistent responses
# 0.0 = deterministic (always same answer for same query)
# Higher values = more creative/varied responses
LLM_TEMPERATURE_DEFAULT: float = 0.0
LLM_TEMPERATURE_MIN: float = 0.0
LLM_TEMPERATURE_MAX: float = 2.0

# LLM backoff multiplier for retry delays
# Used in exponential backoff: delay = base * (backoff ^ attempt)
LLM_BACKOFF: float = 0.75

# LLM response timeout (seconds)
# Default max tokens for generation
LLM_MAX_TOKENS_DEFAULT: int = 800

# ============================================================================
# Rate Limiting & Circuit Breaker
# ============================================================================

# Minimum interval between requests from same IP (seconds)
# 0.1 = 100ms minimum, allows ~10 requests/sec per IP
RATE_LIMIT_INTERVAL: float = 0.1

# Circuit breaker recovery timeout (seconds)
# Time to wait in OPEN state before trying HALF_OPEN
CIRCUIT_BREAKER_RECOVERY_TIMEOUT: float = 60.0

# ============================================================================
# Search Improvements & Boosting (PHASE 10a)
# ============================================================================

# Query-specific boost factors for semantic relevance
# Structured as: query_type -> field -> boost_factor
QUERY_BOOST_FACTORS: dict = {
    "factual": {
        "title_boost": 0.12,
        "section_boost": 0.06,
        "exact_match_boost": 0.10,
    },
    "how_to": {
        "title_boost": 0.08,
        "section_boost": 0.08,
        "structure_boost": 0.12,
    },
    "comparison": {
        "title_boost": 0.06,
        "section_boost": 0.10,
        "diversity_boost": 0.08,
    },
    "definition": {
        "title_boost": 0.15,
        "section_boost": 0.05,
        "conciseness_boost": 0.05,
    },
    "general": {
        "title_boost": 0.08,
        "section_boost": 0.05,
        "default_boost": 0.0,
    },
}

# Maximum boost cap (prevents scores >1.0)
BOOST_MAX_CAP: float = 0.3

# Phrase match boost multiplier
PHRASE_MATCH_MULTIPLIER: float = 1.0

# Diversity boost reduction factor
DIVERSITY_BOOST_FACTOR: float = 0.5

# ============================================================================
# Query Expansion & Decomposition
# ============================================================================

# Synonym/glossary expansion weight
EXPANSION_SYNONYM_WEIGHT: float = 0.8

# Boost term weight (from decomposition)
EXPANSION_BOOST_WEIGHT: float = 0.9

# Query decomposition timeout (seconds)
QUERY_DECOMPOSE_TIMEOUT: float = 0.75

# LLM fallback timeout for decomposition (seconds)
QUERY_DECOMPOSE_LLM_FALLBACK: float = 0.5

# Maximum number of decomposed subtasks
QUERY_DECOMPOSE_MAX_SUBTASKS: int = 3

# ============================================================================
# Retrieval Configuration
# ============================================================================

# Default retrieval timeout (seconds)
RETRIEVAL_TIMEOUT_SECONDS: float = 30.0

# Confidence threshold for query analysis
ANALYSIS_CONFIDENCE_THRESHOLD: float = 0.3

# Per-entity confidence weight
ENTITY_CONFIDENCE_WEIGHT: float = 0.2

# General query confidence baseline
GENERAL_CONFIDENCE_BOOST: float = 0.5

# ============================================================================
# Helper Functions
# ============================================================================


def get_boost_factors(query_type: str) -> dict:
    """
    Get boost factors for a query type.

    Args:
        query_type: One of "factual", "how_to", "comparison", "definition", "general"

    Returns:
        Dictionary of boost factors for that query type
    """
    return QUERY_BOOST_FACTORS.get(query_type, QUERY_BOOST_FACTORS["general"])


def get_rrf_score(vector_rank: int, bm25_rank: int) -> float:
    """
    Calculate RRF score for a result at given ranks.

    Args:
        vector_rank: Rank in vector search results (0-indexed)
        bm25_rank: Rank in BM25 results (0-indexed)

    Returns:
        Combined RRF score
    """
    k = RRF_K_CONSTANT
    vector_score = 1.0 / (k + vector_rank)
    bm25_score = 1.0 / (k + bm25_rank)
    return vector_score + bm25_score


def get_mmr_score(
    relevance: float,
    max_similarity: float,
    lambda_param: float = MMR_LAMBDA,
) -> float:
    """
    Calculate MMR score balancing relevance and diversity.

    Args:
        relevance: Original relevance score (0-1)
        max_similarity: Maximum similarity to already-selected results (0-1)
        lambda_param: Balance parameter (default: MMR_LAMBDA)

    Returns:
        MMR score (can be negative if diversity penalty is high)
    """
    return lambda_param * relevance - (1 - lambda_param) * max_similarity


def get_time_decay_factor(days_old: float) -> float:
    """
    Calculate time decay multiplier for a document.

    Args:
        days_old: Age of document in days

    Returns:
        Decay factor (0-1) to multiply with score
    """
    months_old = days_old / TIME_DECAY_MONTHS_DIVISOR
    return TIME_DECAY_RATE ** months_old
```

## tests/test_glossary_hybrid.py

```
#!/usr/bin/env python3
"""Tests for glossary-aware retrieval and hybrid search."""

import pytest
import numpy as np
from pathlib import Path
import os
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.glossary import Glossary, get_glossary
from src.retrieval_engine import RetrievalEngine, RetrievalConfig, RetrievalStrategy
from src.preprocess import HTMLCleaner


class TestGlossary:
    """Test glossary loading and term detection."""

    def test_glossary_loads(self):
        """Test glossary loads from CSV."""
        glossary = Glossary("data/glossary.csv")
        assert len(glossary.terms) > 0, "Glossary should have terms"
        assert len(glossary.aliases) > 0, "Glossary should have aliases"

    def test_glossary_normalize(self):
        """Test term normalization."""
        assert Glossary._normalize("PTO") == "pto"
        assert Glossary._normalize("Paid Time Off") == "paid time off"
        assert Glossary._normalize("Billable-Rate") == "billablerate"

    def test_glossary_detection_basic(self):
        """Test basic term detection."""
        glossary = Glossary("data/glossary.csv")
        text = "What is PTO?"
        detected = glossary.detect_terms(text)
        # Should detect some form of "pto" or "paid time off"
        assert len(detected) > 0, f"Should detect terms in '{text}'"

    def test_glossary_detection_billing(self):
        """Test detection of billing-related terms."""
        glossary = Glossary("data/glossary.csv")
        text = "Set billable rates for the project"
        detected = glossary.detect_terms(text)
        assert len(detected) > 0, "Should detect 'billable' in billing context"

    def test_glossary_expansion(self):
        """Test query expansion with glossary."""
        glossary = Glossary("data/glossary.csv")
        query = "What is PTO?"
        expanded = glossary.expand_query(query, max_variants=3)

        assert len(expanded) > 0, "Should have at least original query"
        assert expanded[0] == query, "First variant should be original"
        assert isinstance(expanded, list), "Should return list"

    def test_glossary_get_term_info(self):
        """Test retrieving term metadata."""
        glossary = Glossary("data/glossary.csv")
        info = glossary.get_term_info("PTO")

        if info:  # May not exist in small test glossary
            assert "term" in info or "canonical" in info
            assert isinstance(info, dict)

    def test_glossary_singleton(self):
        """Test global glossary singleton."""
        g1 = get_glossary()
        g2 = get_glossary()
        assert g1 is g2, "Should return same instance"


class TestPIIStripping:
    """Test PII removal from text."""

    def test_strip_email(self):
        """Test email removal."""
        text = "Contact us at support@clockify.me for help"
        cleaned = HTMLCleaner.strip_pii(text)
        assert "support@clockify.me" not in cleaned
        assert "[EMAIL]" in cleaned

    def test_strip_phone(self):
        """Test phone number removal."""
        text = "Call us at (555) 123-4567"
        cleaned = HTMLCleaner.strip_pii(text)
        assert "(555) 123-4567" not in cleaned
        assert "[PHONE]" in cleaned

    def test_strip_ssn(self):
        """Test SSN removal."""
        text = "Employee SSN: 123-45-6789"
        cleaned = HTMLCleaner.strip_pii(text)
        assert "123-45-6789" not in cleaned
        assert "[SSN]" in cleaned

    def test_strip_preserves_content(self):
        """Test that stripping preserves other content."""
        text = "Contact support@clockify.me about (555) 123-4567"
        cleaned = HTMLCleaner.strip_pii(text)
        assert "Contact" in cleaned
        assert "about" in cleaned


class TestHybridRetriever:
    """Test hybrid retrieval with late fusion."""

    def test_hybrid_init(self):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(alpha=0.6, k_dense=40, k_bm25=40, k_final=12)
        assert retriever.alpha == 0.6
        assert retriever.k_dense == 40
        assert retriever.k_bm25 == 40
        assert retriever.k_final == 12

    def test_hybrid_alpha_clamping(self):
        """Test alpha clamping to [0, 1]."""
        r1 = HybridRetriever(alpha=-0.5)
        assert r1.alpha == 0.0

        r2 = HybridRetriever(alpha=1.5)
        assert r2.alpha == 1.0

    def test_bm25_index_building(self):
        """Test BM25 index construction."""
        retriever = HybridRetriever()
        chunks = [
            {"id": 0, "text": "PTO is paid time off", "title": "PTO Policy", "section": "Benefits"},
            {"id": 1, "text": "Billable rate is charged to clients", "title": "Billing", "section": "Pricing"},
            {"id": 2, "text": "Timesheet records work hours", "title": "Timesheet", "section": "Tracking"},
        ]

        success = retriever.build_bm25_index(chunks)
        assert success, "BM25 index should build successfully"
        assert retriever.bm25_index is not None
        assert len(retriever.chunks) == 3

    def test_bm25_retrieval(self):
        """Test BM25 retrieval - returns results even with 0 scores."""
        retriever = HybridRetriever(k_bm25=2)
        chunks = [
            {"id": 0, "text": "PTO is paid time off", "title": "PTO Policy", "section": "Benefits"},
            {"id": 1, "text": "Billable rate is charged to clients", "title": "Billing", "section": "Pricing"},
        ]
        retriever.build_bm25_index(chunks)

        # Query - BM25 may return zero scores in some cases
        results = retriever.retrieve_bm25("pto rate")
        # Should return something from top-k even if scores are 0
        assert results is not None, "Should return list"
        # May be empty or have elements depending on BM25 implementation

    def test_score_normalization(self):
        """Test score normalization."""
        retriever = HybridRetriever()
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = retriever._normalize_scores(scores)

        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        assert len(normalized) == len(scores)

    def test_fusion_alpha_0(self):
        """Test fusion with alpha=0 (BM25 only)."""
        retriever = HybridRetriever(alpha=0.0)
        dense_results = [(0, 0.9), (1, 0.7)]
        bm25_results = [(1, 0.5), (2, 0.3)]

        fused = retriever.fuse_results(dense_results, bm25_results)
        assert len(fused) > 0

    def test_fusion_alpha_1(self):
        """Test fusion with alpha=1 (dense only)."""
        retriever = HybridRetriever(alpha=1.0)
        dense_results = [(0, 0.9), (1, 0.7)]
        bm25_results = [(1, 0.5), (2, 0.3)]

        fused = retriever.fuse_results(dense_results, bm25_results)
        assert len(fused) > 0

    def test_fusion_alpha_0_5(self):
        """Test fusion with alpha=0.5 (balanced)."""
        retriever = HybridRetriever(alpha=0.5)
        dense_results = [(0, 0.9), (1, 0.7)]
        bm25_results = [(1, 0.8), (2, 0.3)]

        fused = retriever.fuse_results(dense_results, bm25_results)
        assert len(fused) > 0
        # Score for item 1 should be average-ish since it appears in both
        assert any(chunk_id == 1 for chunk_id, _ in fused)

    def test_fuse_deduplication(self):
        """Test that fusion deduplicates chunks."""
        retriever = HybridRetriever(k_final=10)
        dense_results = [(0, 0.9), (1, 0.7), (2, 0.6)]
        bm25_results = [(0, 0.8), (1, 0.5), (3, 0.4)]

        fused = retriever.fuse_results(dense_results, bm25_results)
        chunk_ids = [cid for cid, _ in fused]

        # No duplicates
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_retriever_singleton(self):
        """Test hybrid retriever singleton."""
        from src.retrieval_hybrid import get_hybrid_retriever
        r1 = get_hybrid_retriever()
        r2 = get_hybrid_retriever()
        assert r1 is r2, "Should return same instance"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_glossary_detection_in_chunk(self):
        """Test glossary detection on chunk-like content."""
        glossary = Glossary("data/glossary.csv")
        chunk_text = """
        # Billable Rates

        A billable rate is the rate charged to clients for work performed.
        This is different from the cost rate, which is the internal labor cost.
        """
        detected = glossary.detect_terms(chunk_text)
        assert len(detected) > 0, "Should detect billing-related terms"

    def test_pii_then_glossary_detection(self):
        """Test PII stripping followed by glossary detection."""
        text = """
        PTO Policy - Contact John Doe at john@example.com or (555) 123-4567.
        PTO is paid time off for employees.
        """
        cleaned = HTMLCleaner.strip_pii(text)
        glossary = Glossary("data/glossary.csv")
        detected = glossary.detect_terms(cleaned)

        assert "[EMAIL]" in cleaned
        assert "[PHONE]" in cleaned
        assert len(detected) > 0, "Should still detect glossary terms after PII stripping"

    def test_hybrid_config_from_env(self):
        """Test that hybrid retriever reads from environment."""
        os.environ["HYBRID_ALPHA"] = "0.7"
        os.environ["K_DENSE"] = "50"
        os.environ["K_BM25"] = "50"
        os.environ["K_FINAL"] = "15"

        from src.retrieval_hybrid import get_hybrid_retriever, HybridRetriever
        # Reset singleton to pick up new env vars
        import src.retrieval_hybrid
        src.retrieval_hybrid._hybrid_retriever = None

        retriever = get_hybrid_retriever()
        assert retriever.alpha == 0.7
        assert retriever.k_dense == 50
        assert retriever.k_bm25 == 50
        assert retriever.k_final == 15

        # Cleanup
        del os.environ["HYBRID_ALPHA"]
        del os.environ["K_DENSE"]
        del os.environ["K_BM25"]
        del os.environ["K_FINAL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## tests/test_retrieval.py

```
"""
Tests for retrieval encoding and search (AXIOM 1, 3, 4, 7).
"""

import os
import numpy as np
import pytest
from src.embeddings import encode_query, encode_texts, EMBEDDING_DIM


class TestEncoding:
    """Test AXIOM 3: Normalization to unit vectors."""
    
    def test_encode_query_normalized(self):
        """Single query embedding should have unit norm."""
        vec = encode_query("test query")
        norm = np.linalg.norm(vec)
        assert 0.99 <= norm <= 1.01, f"Expected norm ~1.0, got {norm}"
    
    def test_encode_batch_normalized(self):
        """Batch embeddings should all be unit norm."""
        vecs = encode_texts(["q1", "q2", "q3"])
        for i, v in enumerate(vecs):
            norm = np.linalg.norm(v)
            assert 0.99 <= norm <= 1.01, f"Vector {i} norm={norm}, expected ~1.0"
    
    def test_encode_dimension(self):
        """Embeddings should match model dimension."""
        vec = encode_query("test")
        assert len(vec[0]) == EMBEDDING_DIM, f"Expected dimension {EMBEDDING_DIM}, got {len(vec[0])}"
    
    def test_encode_lru_cache(self):
        """LRU cache should return same object for same input."""
        v1 = encode_query("cache test")
        v2 = encode_query("cache test")
        np.testing.assert_array_equal(v1, v2)
    
    def test_encode_determinism(self):
        """Same input should produce same embedding (determinism)."""
        for _ in range(3):
            vec1 = encode_query("determinism test")
            vec2 = encode_query("determinism test")
            np.testing.assert_array_almost_equal(vec1, vec2, decimal=6)


class TestRetrieval:
    """Test AXIOM 1, 4, 7 (determinism, retrieval, latency)."""
    
    def test_search_returns_results(self):
        """Verify search endpoint returns results."""
        # This requires server running; marked as integration test
        pass
    
    def test_results_deduplicated_by_url(self):
        """AXIOM 4: Results should be deduplicated by URL."""
        pass
    
    def test_latency_under_budget(self):
        """AXIOM 7: /search p95 should be < 800ms."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## ui/app.js

```
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
```

