// Production RAG Studio — Apple HIG Interface Logic

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const queryInput = document.getElementById('query-input');
  const btnExecute = document.getElementById('btn-execute-query');
  const chatTimeline = document.getElementById('chat-timeline');
  const btnNewChat = document.getElementById('btn-new-chat');
  const sessionBadge = document.getElementById('session-badge');
  const emptyState = document.getElementById('empty-state');
  const telemetryHud = document.getElementById('telemetry-hud');

  // Conversational Session State
  let currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 9);
  if (sessionBadge) sessionBadge.textContent = `ID: ${currentSessionId}`;

  // Telemetry HUD items
  const hudRetrievalTime = document.getElementById('hud-retrieval-time');
  const hudTotalTime = document.getElementById('hud-total-time');
  const hudFaithfulness = document.getElementById('hud-faithfulness');
  const hudPrecision = document.getElementById('hud-precision');
  const hudConfidence = document.getElementById('hud-confidence');

  // Inspector
  const inspectorChunksList = document.getElementById('inspector-chunks-list');
  const inspectorCount = document.getElementById('inspector-count');

  // Controls
  const inputAlpha = document.getElementById('input-alpha');
  const labelAlpha = document.getElementById('label-alpha');
  const inputRerankK = document.getElementById('input-rerank-k');
  const labelRerankK = document.getElementById('label-rerank-k');

  // Tabs
  const segmentBtns = document.querySelectorAll('.segment-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  // Eval Elements
  const btnRunFullEval = document.getElementById('btn-run-full-eval');
  const scRecall = document.getElementById('sc-recall');
  const scMrr = document.getElementById('sc-mrr');
  const scFaithfulness = document.getElementById('sc-faithfulness');
  const scPrecision = document.getElementById('sc-precision');
  const scLatency = document.getElementById('sc-latency');
  const scCost = document.getElementById('sc-cost');
  const benchmarkTableBody = document.getElementById('benchmark-table-body');

  // Ingest Modal Elements
  const btnOpenIngest = document.getElementById('btn-open-ingest');
  const ingestModal = document.getElementById('ingest-modal');
  const btnCloseModal = document.getElementById('btn-close-modal');
  const btnCancelIngest = document.getElementById('btn-cancel-ingest');
  const btnSubmitIngest = document.getElementById('btn-submit-ingest');
  const btnCopyAnswer = document.getElementById('btn-copy-answer');

  // Settings Modal Elements
  const btnOpenSettings = document.getElementById('btn-open-settings');
  const settingsModal = document.getElementById('settings-modal');
  const btnCloseSettings = document.getElementById('btn-close-settings');
  const btnSaveSettings = document.getElementById('btn-save-settings');
  const btnTestConnection = document.getElementById('btn-test-connection');
  const settingBaseUrl = document.getElementById('setting-base-url');
  const settingApiKey = document.getElementById('setting-api-key');
  const settingLlmModel = document.getElementById('setting-llm-model');
  const settingEmbeddingModel = document.getElementById('setting-embedding-model');
  const connectionStatusMsg = document.getElementById('connection-status-msg');
  const gatewayStatusPill = document.getElementById('gateway-status-pill');
  const pipelineStepper = document.getElementById('pipeline-stepper');
  const traceGatewayLabel = document.getElementById('trace-gateway-label');

  // Suggestion chips
  document.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      queryInput.value = chip.getAttribute('data-query');
      executeQuery();
    });
  });

  // Slider controls
  inputAlpha.addEventListener('input', (e) => {
    const val = e.target.value;
    const denseWeight = (val / 50).toFixed(1);
    const sparseWeight = ((100 - val) / 50).toFixed(1);
    labelAlpha.textContent = `${denseWeight} : ${sparseWeight}`;
  });

  inputRerankK.addEventListener('input', (e) => {
    labelRerankK.textContent = e.target.value;
  });

  // Tab switching
  segmentBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      segmentBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.style.display = 'none');

      btn.classList.add('active');
      const targetTab = document.getElementById(btn.getAttribute('data-tab'));
      if (targetTab) {
        targetTab.style.display = 'block';
      }

      if (btn.getAttribute('data-tab') === 'tab-evals') {
        loadMetrics();
      }
      if (btn.getAttribute('data-tab') === 'tab-support') {
        loadSupportCustomer();
      }
    });
  });

  // Execute Query
  // Helper to escape HTML characters
  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Reset / New Chat Handler
  if (btnNewChat) {
    btnNewChat.addEventListener('click', async () => {
      const oldSession = currentSessionId;
      currentSessionId = 'sess_' + Math.random().toString(36).substring(2, 9);
      if (sessionBadge) sessionBadge.textContent = `ID: ${currentSessionId}`;
      if (chatTimeline) {
        chatTimeline.innerHTML = '';
        chatTimeline.style.display = 'none';
      }
      emptyState.style.display = 'flex';
      telemetryHud.style.display = 'none';
      try {
        await fetch(`/api/chat/reset/${oldSession}`, { method: 'POST' });
      } catch (e) {
        console.warn('Could not reset session:', e);
      }
    });
  }

  // Execute Conversational RAG Query
  async function executeQuery() {
    const query = queryInput.value.trim();
    if (!query) return;

    btnExecute.disabled = true;
    btnExecute.innerHTML = '<span>Thinking...</span>';

    // 1. Show chat timeline & hide empty state
    emptyState.style.display = 'none';
    chatTimeline.style.display = 'flex';
    telemetryHud.style.display = 'flex';

    // 2. Append User Message Bubble
    const userRow = document.createElement('div');
    userRow.className = 'chat-row user-row';
    userRow.innerHTML = `
      <div class="chat-bubble-user">
        ${escapeHtml(query)}
      </div>
    `;
    chatTimeline.appendChild(userRow);

    // 3. Clear user input
    queryInput.value = '';

    // 4. Append Typing / Shimmer Indicator
    const typingRow = document.createElement('div');
    typingRow.className = 'chat-row assistant-row typing-row';
    typingRow.id = 'typing-indicator';
    typingRow.innerHTML = `
      <div class="typing-bubble">
        <div class="typing-dots"><span></span><span></span><span></span></div>
        <span>Retrieving chunks & verifying citations...</span>
      </div>
    `;
    chatTimeline.appendChild(typingRow);
    chatTimeline.scrollTop = chatTimeline.scrollHeight;

    const alphaVal = parseInt(inputAlpha.value, 10);
    const denseWeight = alphaVal / 50;
    const sparseWeight = (100 - alphaVal) / 50;
    const rerankK = parseInt(inputRerankK.value, 10);

    try {
      let response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSessionId,
          message: query,
          dense_top_k: 15,
          sparse_top_k: 15,
          rerank_top_k: rerankK,
          dense_weight: denseWeight,
          sparse_weight: sparseWeight,
        }),
      });

      // Fallback to /api/query if backend server was started before /api/chat was registered
      if (response.status === 404 || response.status === 405) {
        response = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            dense_top_k: 15,
            sparse_top_k: 15,
            rerank_top_k: rerankK,
            dense_weight: denseWeight,
            sparse_weight: sparseWeight,
          }),
        });
      }

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      if (!data.original_message) {
        data.original_message = query;
      }
      
      // Remove typing indicator
      const typingEl = document.getElementById('typing-indicator');
      if (typingEl) typingEl.remove();

      // Render Assistant Response Bubble
      renderAssistantMessage(data);

    } catch (err) {
      console.error('Chat error:', err);
      const typingEl = document.getElementById('typing-indicator');
      if (typingEl) typingEl.remove();

      const errRow = document.createElement('div');
      errRow.className = 'chat-row assistant-row';
      errRow.innerHTML = `
        <div class="chat-card-assistant" style="border-color: rgba(255, 69, 58, 0.4);">
          <div style="color: var(--accent-red); font-size: 13px;">Error: ${err.message}. Please check backend connection.</div>
        </div>
      `;
      chatTimeline.appendChild(errRow);
      chatTimeline.scrollTop = chatTimeline.scrollHeight;
    } finally {
      btnExecute.disabled = false;
      btnExecute.innerHTML = '<span>Send</span><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
    }
  }

  btnExecute.addEventListener('click', executeQuery);
  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') executeQuery();
  });

  // Render Assistant Message with footnotes, citations, and metrics
  function renderAssistantMessage(data) {
    // 1. Update Telemetry HUD
    hudRetrievalTime.textContent = `${data.retrieval_latency_ms} ms`;
    hudTotalTime.textContent = `${data.total_latency_ms} ms`;
    hudFaithfulness.textContent = `${(data.faithfulness_score * 100).toFixed(0)}%`;
    hudPrecision.textContent = `${(data.citation_precision * 100).toFixed(0)}%`;
    hudConfidence.textContent = data.confidence_score.toFixed(2);

    // 2. Format answer text with interactive footnote badges
    let formattedAnswer = escapeHtml(data.answer);
    formattedAnswer = formattedAnswer.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<span class="footnote-cite" data-cite-idx="${p1}">[${p1}]</span>`;
    });

    const assistantRow = document.createElement('div');
    assistantRow.className = 'chat-row assistant-row';

    // Query reformulation pill if pronoun or follow-up was rewritten
    let reformulatedHtml = '';
    if (data.rewritten_query && data.rewritten_query.trim().toLowerCase() !== data.original_message.trim().toLowerCase()) {
      reformulatedHtml = `
        <div class="reformulated-badge" title="Query condenser resolved context and pronouns for retrieval">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          <span>Standalone query: <strong>${escapeHtml(data.rewritten_query)}</strong></span>
        </div>
      `;
    }

    // Provenance Citation Cards
    let citationsHtml = '';
    if (data.citations && data.citations.length > 0) {
      citationsHtml = '<div class="citations-section"><div class="citations-title">Verified Citations & Provenance</div><div class="citations-grid">';
      data.citations.forEach(cit => {
        citationsHtml += `
          <div class="citation-card" id="cit-card-${cit.citation_index}">
            <div class="citation-card-header">
              <span class="citation-source">[${cit.citation_index}] ${escapeHtml(cit.source_title)} • ${escapeHtml(cit.section_title)}</span>
              <span class="citation-grounded-pill">${cit.is_grounded ? '✓ Grounded' : '⚠ Ungrounded'}</span>
            </div>
            <div class="citation-quote">"${escapeHtml(cit.quote)}"</div>
          </div>
        `;
      });
      citationsHtml += '</div></div>';
    }

    assistantRow.innerHTML = `
      <div class="chat-card-assistant">
        ${reformulatedHtml}
        <div class="answer-body">
          ${formattedAnswer}
        </div>
        ${citationsHtml}
        <div class="chat-card-footer">
          <div class="chat-card-metrics">
            <span>⏱ ${data.total_latency_ms} ms</span>
            <span style="color: var(--accent-green);">✓ ${(data.faithfulness_score * 100).toFixed(0)}% Faithfulness</span>
            <span style="color: var(--accent-blue);">🎯 ${(data.citation_precision * 100).toFixed(0)}% Precision</span>
          </div>
          <button class="apple-button icon btn-copy-turn" title="Copy answer">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          </button>
        </div>
      </div>
    `;

    chatTimeline.appendChild(assistantRow);
    chatTimeline.scrollTop = chatTimeline.scrollHeight;

    // Connect copy button
    const copyBtn = assistantRow.querySelector('.btn-copy-turn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(data.answer).then(() => {
          copyBtn.innerHTML = '<span style="font-size: 11px; color: var(--accent-green);">✓</span>';
          setTimeout(() => {
            copyBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
          }, 1500);
        });
      });
    }

    // Connect Footnotes to Inspector
    assistantRow.querySelectorAll('.footnote-cite').forEach(el => {
      el.addEventListener('click', () => {
        const idx = el.getAttribute('data-cite-idx');
        highlightCitation(idx);
      });
    });

    // Update Inspector Chunks & Pipeline Trace for latest turn
    renderInspectorChunks(data.retrieved_chunks);
    if (data.pipeline_trace) {
      renderPipelineTrace(data.pipeline_trace);
    }
  }

  function renderPipelineTrace(trace) {
    if (!trace || trace.length === 0) return;
    pipelineStepper.innerHTML = '';

    trace.forEach((stage, idx) => {
      const item = document.createElement('div');
      item.className = 'timeline-item';

      let detailsHtml = '';
      if (stage.details) {
        for (const [key, val] of Object.entries(stage.details)) {
          detailsHtml += `<span class="stage-tag">${key.replace(/_/g, ' ')}: <b>${val}</b></span>`;
        }
      }

      item.innerHTML = `
        <div class="timeline-badge">${idx + 1}</div>
        <div class="timeline-card">
          <div class="timeline-card-header">
            <span class="stage-title">${stage.stage}</span>
            <span class="stage-latency">${stage.latency_ms} ms</span>
          </div>
          <div class="stage-details">${detailsHtml}</div>
        </div>
      `;
      pipelineStepper.appendChild(item);
    });
  }

  function highlightCitation(idx) {
    // Highlight citation card
    document.querySelectorAll('.citation-card').forEach(c => c.style.borderColor = 'var(--glass-border)');
    const targetCitCard = document.getElementById(`cit-card-${idx}`);
    if (targetCitCard) {
      targetCitCard.style.borderColor = 'var(--accent-blue)';
      targetCitCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Highlight inspector chunk
    document.querySelectorAll('.chunk-card').forEach(c => c.classList.remove('active-highlight'));
    const chunkCard = document.getElementById(`chunk-card-${idx}`);
    if (chunkCard) {
      chunkCard.classList.add('active-highlight');
      chunkCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  function renderInspectorChunks(chunks) {
    inspectorChunksList.innerHTML = '';
    inspectorCount.textContent = `${chunks.length} chunks`;

    if (!chunks || chunks.length === 0) {
      inspectorChunksList.innerHTML = '<div class="inspector-empty">No candidate chunks retrieved.</div>';
      return;
    }

    chunks.forEach((chunk, i) => {
      const idx = i + 1;
      const card = document.createElement('div');
      card.className = 'chunk-card';
      card.id = `chunk-card-${idx}`;

      const denseRank = chunk.dense_rank || 'N/A';
      const sparseRank = chunk.sparse_rank || 'N/A';
      const rrfScore = chunk.rrf_score ? chunk.rrf_score.toFixed(4) : (chunk.score ? chunk.score.toFixed(4) : 'N/A');

      card.innerHTML = `
        <div class="chunk-header">
          <span class="chunk-title">${chunk.source_title}</span>
          <span class="chunk-rank-pill">#${idx} RRF: ${rrfScore}</span>
        </div>
        <div class="chunk-ranks-grid">
          <div>Dense: #${denseRank}</div>
          <div>BM25: #${sparseRank}</div>
          <div>${chunk.token_count || 120} tok</div>
        </div>
        <div class="chunk-text">${chunk.content}</div>
      `;
      inspectorChunksList.appendChild(card);
    });
  }

  // Load and Render Evaluation Scorecard
  async function loadMetrics() {
    try {
      const res = await fetch('/api/metrics');
      if (!res.ok) return;
      const data = await res.json();
      const sc = data.scorecard;

      if (sc) {
        scRecall.textContent = `${(sc.recall_at_5 * 100).toFixed(1)}%`;
        scMrr.textContent = sc.mrr.toFixed(3);
        scFaithfulness.textContent = `${(sc.mean_faithfulness * 100).toFixed(1)}%`;
        scPrecision.textContent = `${(sc.mean_citation_precision * 100).toFixed(1)}%`;
        scLatency.textContent = `${sc.p95_latency_ms} ms`;
        scCost.textContent = `$${sc.cost_per_1k_queries_usd.toFixed(4)}`;
      }

      if (data.runs) {
        benchmarkTableBody.innerHTML = '';
        data.runs.forEach(run => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td class="mono">${run.id}</td>
            <td>${run.question}</td>
            <td class="mono">${run.target_doc_id || 'Negative Control'}</td>
            <td><span class="badge-status" style="background: rgba(52,199,89,0.15); color: var(--accent-green);">${(run.recall_at_5 * 100).toFixed(0)}%</span></td>
            <td>${(run.faithfulness * 100).toFixed(0)}%</td>
            <td>${run.citations_count}</td>
            <td class="mono">${run.total_latency_ms} ms</td>
          `;
          benchmarkTableBody.appendChild(tr);
        });
      }
    } catch (e) {
      console.warn('Could not fetch metrics:', e);
    }
  }

  btnRunFullEval.addEventListener('click', async () => {
    btnRunFullEval.disabled = true;
    btnRunFullEval.innerHTML = '<span>Running Test Suite...</span>';
    try {
      const res = await fetch('/api/eval/run', { method: 'POST' });
      if (res.ok) {
        await loadMetrics();
      }
    } catch (e) {
      console.error('Eval run error:', e);
    } finally {
      btnRunFullEval.disabled = false;
      btnRunFullEval.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg><span>Run Automated Benchmark</span>';
    }
  });

  // Modal Handling
  btnOpenIngest.addEventListener('click', () => {
    ingestModal.style.display = 'flex';
  });

  const closeModal = () => { ingestModal.style.display = 'none'; };
  btnCloseModal.addEventListener('click', closeModal);
  btnCancelIngest.addEventListener('click', closeModal);
  ingestModal.addEventListener('click', (e) => {
    if (e.target === ingestModal) closeModal();
  });

  if (btnCopyAnswer) {
    btnCopyAnswer.addEventListener('click', () => {
      const text = answerText.innerText || answerText.textContent;
      if (!text) return;
      navigator.clipboard.writeText(text).then(() => {
        const prev = btnCopyAnswer.innerHTML;
        btnCopyAnswer.innerHTML = '<span style="font-size: 11px; color: var(--accent-green);">✓</span>';
        setTimeout(() => { btnCopyAnswer.innerHTML = prev; }, 1500);
      });
    });
  }

  btnSubmitIngest.addEventListener('click', async () => {
    const docId = document.getElementById('ingest-doc-id').value.trim();
    const title = document.getElementById('ingest-doc-title').value.trim();
    const content = document.getElementById('ingest-doc-content').value.trim();

    if (!docId || !title || !content) {
      alert('Please fill in all document fields.');
      return;
    }

    btnSubmitIngest.disabled = true;
    btnSubmitIngest.textContent = 'Chunking & Indexing...';

    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doc_id: docId, title: title, content: content }),
      });

      if (!res.ok) throw new Error(`Ingest failed: ${res.status}`);
      const data = await res.json();
      alert(`Successfully indexed "${title}" into ${data.chunks_created} chunks!`);
      closeModal();
      // Add to sidebar list
      const sidebarList = document.getElementById('sidebar-doc-list');
      const li = document.createElement('li');
      li.className = 'doc-item';
      li.innerHTML = `
        <span class="doc-icon">📑</span>
        <div class="doc-info">
          <span class="doc-name">${title}</span>
          <span class="doc-meta">${data.chunks_created} chunks</span>
        </div>
      `;
      sidebarList.appendChild(li);
    } catch (e) {
      alert('Error ingesting document: ' + e.message);
    } finally {
      btnSubmitIngest.disabled = false;
      btnSubmitIngest.textContent = 'Chunk & Index';
    }
  });

  // Settings Modal Handlers
  async function loadSettings() {
    try {
      const res = await fetch('/api/settings');
      if (res.ok) {
        const data = await res.json();
        if (data.openai_base_url) {
          settingBaseUrl.value = data.openai_base_url;
          traceGatewayLabel.textContent = `Gateway: ${data.openai_base_url}`;
          if (data.openai_base_url.includes('tokenrouter')) {
            gatewayStatusPill.textContent = 'TokenRouter Live';
          } else {
            gatewayStatusPill.textContent = 'Custom Gateway';
          }
        }
        if (data.llm_model) settingLlmModel.value = data.llm_model;
        if (data.embedding_model) settingEmbeddingModel.value = data.embedding_model;
      }
    } catch (e) {
      console.warn('Could not load settings:', e);
    }
  }

  btnOpenSettings.addEventListener('click', () => {
    connectionStatusMsg.style.display = 'none';
    settingsModal.style.display = 'flex';
  });

  const closeSettings = () => { settingsModal.style.display = 'none'; };
  btnCloseSettings.addEventListener('click', closeSettings);
  settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) closeSettings();
  });

  btnSaveSettings.addEventListener('click', async () => {
    btnSaveSettings.disabled = true;
    btnSaveSettings.textContent = 'Saving...';

    const payload = {
      openai_base_url: settingBaseUrl.value.trim(),
      openai_api_key: settingApiKey.value.trim() || null,
      llm_model: settingLlmModel.value.trim(),
      embedding_model: settingEmbeddingModel.value.trim(),
    };

    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        await loadSettings();
        connectionStatusMsg.style.display = 'block';
        connectionStatusMsg.style.background = 'rgba(52, 199, 89, 0.15)';
        connectionStatusMsg.style.color = 'var(--accent-green)';
        connectionStatusMsg.textContent = '✓ Settings updated and applied to active pipeline!';
        setTimeout(closeSettings, 1000);
      }
    } catch (e) {
      alert('Error updating settings: ' + e.message);
    } finally {
      btnSaveSettings.disabled = false;
      btnSaveSettings.textContent = 'Save & Apply';
    }
  });

  btnTestConnection.addEventListener('click', async () => {
    btnTestConnection.disabled = true;
    btnTestConnection.textContent = 'Testing...';
    connectionStatusMsg.style.display = 'block';
    connectionStatusMsg.style.background = 'rgba(255, 255, 255, 0.08)';
    connectionStatusMsg.style.color = 'var(--text-secondary)';
    connectionStatusMsg.textContent = 'Pinging gateway at ' + settingBaseUrl.value + '...';

    try {
      const res = await fetch('/api/settings/test', { method: 'POST' });
      const data = await res.json();
      if (data.status === 'connected') {
        connectionStatusMsg.style.background = 'rgba(52, 199, 89, 0.15)';
        connectionStatusMsg.style.color = 'var(--accent-green)';
        connectionStatusMsg.textContent = `✓ Gateway Reachable: ${data.message} (Status: ${data.status_code})`;
      } else {
        connectionStatusMsg.style.background = 'rgba(255, 159, 10, 0.15)';
        connectionStatusMsg.style.color = 'var(--accent-orange)';
        connectionStatusMsg.textContent = `Gateway Response: ${data.message}`;
      }
    } catch (e) {
      connectionStatusMsg.style.background = 'rgba(255, 69, 58, 0.15)';
      connectionStatusMsg.style.color = 'var(--accent-red)';
      connectionStatusMsg.textContent = 'Connection failed: ' + e.message;
    } finally {
      btnTestConnection.disabled = false;
      btnTestConnection.textContent = 'Test Gateway Ping';
    }
  });

  // ─── E-Commerce Support Chat Controller ─────────────────────────────
  const supportCustomerName = document.getElementById('support-customer-name');
  const supportTierBadge = document.getElementById('support-tier-badge');
  const supportPoints = document.getElementById('support-points');
  const supportEmail = document.getElementById('support-email');
  const supportOrdersCount = document.getElementById('support-orders-count');
  const btnNewSupportChat = document.getElementById('btn-new-support-chat');
  const supportTelemetryHud = document.getElementById('support-telemetry-hud');
  const supportHudRetrieval = document.getElementById('support-hud-retrieval');
  const supportHudTotal = document.getElementById('support-hud-total');
  const supportHudFaith = document.getElementById('support-hud-faith');
  const supportHudConfidence = document.getElementById('support-hud-confidence');
  const supportEmptyState = document.getElementById('support-empty-state');
  const supportChatTimeline = document.getElementById('support-chat-timeline');
  const supportQueryInput = document.getElementById('support-query-input');
  const btnSupportSend = document.getElementById('btn-support-send');

  let currentSupportSessionId = 'support_' + Math.random().toString(36).substring(2, 9);
  let customerDataLoaded = false;

  async function loadSupportCustomer() {
    try {
      const res = await fetch('/api/support/customer');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const customer = data.customer || {};
      const orders = data.orders || [];

      if (supportCustomerName) supportCustomerName.textContent = customer.name || 'Demo Customer';
      if (supportTierBadge) supportTierBadge.textContent = customer.tier || 'Member';
      if (supportPoints) supportPoints.textContent = `${(customer.loyalty_points || 0).toLocaleString()} pts`;
      if (supportEmail) supportEmail.textContent = customer.email || '';
      if (supportOrdersCount) supportOrdersCount.textContent = `${orders.length} order${orders.length === 1 ? '' : 's'}`;
      customerDataLoaded = true;
    } catch (err) {
      console.warn('Could not load support customer profile:', err);
    }
  }

  // Support suggestion chips
  document.querySelectorAll('.support-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      if (supportQueryInput) {
        supportQueryInput.value = chip.getAttribute('data-support-query') || '';
        executeSupportQuery();
      }
    });
  });

  // Reset Support Session
  if (btnNewSupportChat) {
    btnNewSupportChat.addEventListener('click', async () => {
      const oldSession = currentSupportSessionId;
      currentSupportSessionId = 'support_' + Math.random().toString(36).substring(2, 9);
      if (supportChatTimeline) {
        supportChatTimeline.innerHTML = '';
        supportChatTimeline.style.display = 'none';
      }
      if (supportEmptyState) supportEmptyState.style.display = 'flex';
      if (supportTelemetryHud) supportTelemetryHud.style.display = 'none';
      try {
        await fetch(`/api/chat/reset/${oldSession}`, { method: 'POST' });
      } catch (e) {
        console.warn('Could not reset support session:', e);
      }
    });
  }

  // Execute Support Query
  async function executeSupportQuery() {
    if (!supportQueryInput || !btnSupportSend) return;
    const query = supportQueryInput.value.trim();
    if (!query) return;

    btnSupportSend.disabled = true;
    btnSupportSend.innerHTML = '<span>Thinking...</span>';

    if (supportEmptyState) supportEmptyState.style.display = 'none';
    if (supportChatTimeline) supportChatTimeline.style.display = 'flex';
    if (supportTelemetryHud) supportTelemetryHud.style.display = 'flex';

    // Append User Bubble
    const userRow = document.createElement('div');
    userRow.className = 'chat-row user-row';
    userRow.innerHTML = `
      <div class="chat-bubble-user">
        ${escapeHtml(query)}
      </div>
    `;
    supportChatTimeline.appendChild(userRow);
    supportQueryInput.value = '';

    // Append Shimmer / Typing Indicator
    const typingRow = document.createElement('div');
    typingRow.className = 'chat-row assistant-row typing-row';
    typingRow.id = 'support-typing-indicator';
    typingRow.innerHTML = `
      <div class="typing-bubble">
        <div class="typing-dots"><span></span><span></span><span></span></div>
        <span>Consulting product catalog & verifying order history...</span>
      </div>
    `;
    supportChatTimeline.appendChild(typingRow);
    supportChatTimeline.scrollTop = supportChatTimeline.scrollHeight;

    try {
      const response = await fetch('/api/support/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSupportSessionId,
          message: query,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      const typingEl = document.getElementById('support-typing-indicator');
      if (typingEl) typingEl.remove();

      renderSupportAssistantMessage(data);
    } catch (err) {
      console.error('Support chat query failed:', err);
      const typingEl = document.getElementById('support-typing-indicator');
      if (typingEl) typingEl.remove();

      const errRow = document.createElement('div');
      errRow.className = 'chat-row assistant-row';
      errRow.innerHTML = `
        <div class="chat-card-assistant" style="border-color: rgba(255, 69, 58, 0.4);">
          <div style="color: var(--accent-red); font-size: 13px;">Error: ${err.message}. Please check backend service.</div>
        </div>
      `;
      supportChatTimeline.appendChild(errRow);
      supportChatTimeline.scrollTop = supportChatTimeline.scrollHeight;
    } finally {
      btnSupportSend.disabled = false;
      btnSupportSend.innerHTML = '<span>Send</span><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>';
    }
  }

  if (btnSupportSend) btnSupportSend.addEventListener('click', executeSupportQuery);
  if (supportQueryInput) {
    supportQueryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') executeSupportQuery();
    });
  }

  // Render Assistant Message for Support
  function renderSupportAssistantMessage(data) {
    if (supportHudRetrieval) supportHudRetrieval.textContent = `${data.retrieval_latency_ms} ms`;
    if (supportHudTotal) supportHudTotal.textContent = `${data.total_latency_ms} ms`;
    if (supportHudFaith) supportHudFaith.textContent = `${(data.faithfulness_score * 100).toFixed(0)}%`;
    if (supportHudConfidence) supportHudConfidence.textContent = data.confidence_score.toFixed(2);

    let formattedAnswer = escapeHtml(data.answer);
    formattedAnswer = formattedAnswer.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<span class="footnote-cite" data-cite-idx="${p1}">[${p1}]</span>`;
    });

    const assistantRow = document.createElement('div');
    assistantRow.className = 'chat-row assistant-row';

    let citationsHtml = '';
    if (data.citations && data.citations.length > 0) {
      citationsHtml = '<div class="citations-section"><div class="citations-title">Verified Catalog Citations & Grounding</div><div class="citations-grid">';
      data.citations.forEach(cit => {
        citationsHtml += `
          <div class="citation-card" id="cit-card-${cit.citation_index}">
            <div class="citation-card-header">
              <span class="citation-source">[${cit.citation_index}] ${escapeHtml(cit.source_title)} • ${escapeHtml(cit.section_title)}</span>
              <span class="citation-grounded-pill">${cit.is_grounded ? '✓ Grounded' : '⚠ Ungrounded'}</span>
            </div>
            <div class="citation-quote">"${escapeHtml(cit.quote)}"</div>
          </div>
        `;
      });
      citationsHtml += '</div></div>';
    }

    assistantRow.innerHTML = `
      <div class="chat-card-assistant">
        <div class="answer-body">
          ${formattedAnswer}
        </div>
        ${citationsHtml}
        <div class="chat-card-footer">
          <div class="chat-card-metrics">
            <span>⏱ ${data.total_latency_ms} ms</span>
            <span style="color: var(--accent-green);">✓ ${(data.faithfulness_score * 100).toFixed(0)}% Faithfulness</span>
            <span style="color: var(--accent-blue);">🎯 ${(data.citation_precision * 100).toFixed(0)}% Precision</span>
          </div>
          <button class="apple-button icon btn-copy-turn" title="Copy answer">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          </button>
        </div>
      </div>
    `;

    supportChatTimeline.appendChild(assistantRow);
    supportChatTimeline.scrollTop = supportChatTimeline.scrollHeight;

    const copyBtn = assistantRow.querySelector('.btn-copy-turn');
    if (copyBtn) {
      copyBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(data.answer).then(() => {
          copyBtn.innerHTML = '<span style="font-size: 11px; color: var(--accent-green);">✓</span>';
          setTimeout(() => {
            copyBtn.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
          }, 1500);
        });
      });
    }

    assistantRow.querySelectorAll('.footnote-cite').forEach(el => {
      el.addEventListener('click', () => {
        const idx = el.getAttribute('data-cite-idx');
        highlightCitation(idx);
      });
    });

    if (data.retrieved_chunks) {
      renderInspectorChunks(data.retrieved_chunks);
    }
    if (data.pipeline_trace) {
      renderPipelineTrace(data.pipeline_trace);
    }
  }

  // Load settings and support customer on startup
  loadSettings();
  loadSupportCustomer();
});
