/* Skadi RF Signal Identification — Dashboard Client */
/* Vanilla JS, no external dependencies */

(function() {
    'use strict';

    // SocketIO connection
    const socket = io();
    let sortColumn = 'timestamp_utc';
    let sortDirection = 'desc';

    // Load data immediately on page load (don't wait for SocketIO)
    console.log('Skadi JS loaded, calling loadHistory...');
    try {
        loadHistory();
        pollStatus();
    } catch(e) {
        console.error('Init error:', e);
    }

    // --- SocketIO Events ---

    socket.on('connect', function() {
        console.log('Connected to Skadi');
        loadHistory();
    });

    socket.on('disconnect', function() {
        console.log('Disconnected from Skadi');
        document.getElementById('scanner-status').textContent = 'OFFLINE';
        document.getElementById('scanner-status').className = 'badge badge-idle';
    });

    socket.on('new_detections', function(data) {
        if (data.detections && data.detections.length > 0) {
            addAlertCards(data.detections);
            loadHistory(); // Refresh history table
        }
    });

    socket.on('scan_status', function(data) {
        updateStatus(data);
    });

    // --- Alert Cards ---

    function addAlertCards(detections) {
        const feed = document.getElementById('alert-feed');
        const empty = feed.querySelector('.empty-state');
        if (empty) empty.remove();

        detections.forEach(function(det) {
            const card = createAlertCard(det);
            feed.insertBefore(card, feed.firstChild);
        });

        // Keep only last 100 alerts
        while (feed.children.length > 100) {
            feed.removeChild(feed.lastChild);
        }
    }

    function createAlertCard(det) {
        const threat = det.threat_level || 'MEDIUM';
        const card = document.createElement('div');
        card.className = 'alert-card threat-' + threat;

        const freqMhz = (det.frequency_hz / 1e6).toFixed(3);
        const conf = det.confidence_score != null ? det.confidence_score.toFixed(2) : '-';
        const power = det.signal_strength_dbm != null ? det.signal_strength_dbm.toFixed(1) : '-';
        const sigType = det.signal_type || 'UNKNOWN';
        const mod = det.modulation || '?';

        card.innerHTML =
            '<span class="alert-threat ' + threat + '">' + threat + '</span>' +
            '<span class="alert-type">' + escapeHtml(sigType) + '</span>' +
            '<span class="alert-freq">' + freqMhz + ' MHz</span>' +
            '<span class="alert-mod">' + escapeHtml(mod) + '</span>' +
            '<span class="alert-conf">Conf: ' + conf + '</span>' +
            '<span class="alert-power">' + power + ' dBm</span>';

        return card;
    }

    // --- History Table ---

    function loadHistory() {
        const params = new URLSearchParams();
        const threatFilter = document.getElementById('filter-threat').value;
        const freqMin = document.getElementById('filter-freq-min').value;
        const freqMax = document.getElementById('filter-freq-max').value;

        if (threatFilter) params.set('threat_level', threatFilter);
        if (freqMin) params.set('freq_min', parseFloat(freqMin) * 1e6);
        if (freqMax) params.set('freq_max', parseFloat(freqMax) * 1e6);
        params.set('limit', '500');

        fetch('/api/detections?' + params.toString())
            .then(function(r) { return r.json(); })
            .then(function(data) { renderHistoryTable(data); })
            .catch(function(err) { console.error('Failed to load history:', err); });
    }

    function renderHistoryTable(rows) {
        // Sort
        rows.sort(function(a, b) {
            let va = a[sortColumn];
            let vb = b[sortColumn];
            if (va == null) va = '';
            if (vb == null) vb = '';
            if (typeof va === 'number' && typeof vb === 'number') {
                return sortDirection === 'asc' ? va - vb : vb - va;
            }
            va = String(va);
            vb = String(vb);
            return sortDirection === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
        });

        const tbody = document.getElementById('history-body');
        tbody.innerHTML = '';

        rows.forEach(function(row) {
            const tr = document.createElement('tr');
            const threat = row.threat_level || 'MEDIUM';
            const freqMhz = row.frequency_hz != null ? (row.frequency_hz / 1e6).toFixed(3) : '-';
            const bwKhz = row.bandwidth_hz != null ? (row.bandwidth_hz / 1e3).toFixed(1) : '-';
            const conf = row.confidence_score != null ? row.confidence_score.toFixed(2) : '-';
            const power = row.signal_strength_dbm != null ? row.signal_strength_dbm.toFixed(1) : '-';
            const time = row.timestamp_utc ? row.timestamp_utc.substring(11, 19) : '-';

            tr.innerHTML =
                '<td>' + time + '</td>' +
                '<td style="font-family:monospace;color:var(--accent)">' + freqMhz + '</td>' +
                '<td>' + escapeHtml(row.modulation || '?') + '</td>' +
                '<td>' + escapeHtml(row.signal_type || 'UNKNOWN') + '</td>' +
                '<td>' + conf + '</td>' +
                '<td><span class="threat-badge ' + threat + '">' + threat + '</span></td>' +
                '<td>' + bwKhz + '</td>' +
                '<td>' + power + '</td>';
            tbody.appendChild(tr);
        });

        // Update sort indicators
        document.querySelectorAll('#history-table th').forEach(function(th) {
            th.classList.remove('sort-asc', 'sort-desc');
            if (th.dataset.sort === sortColumn) {
                th.classList.add(sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
            }
        });
    }

    // --- Status Polling ---

    function updateStatus(data) {
        const badge = document.getElementById('scanner-status');
        if (data.scanning) {
            badge.textContent = 'SCANNING';
            badge.className = 'badge badge-scanning';
        } else {
            badge.textContent = 'IDLE';
            badge.className = 'badge badge-idle';
        }

        if (data.sweep_count != null) {
            document.getElementById('sweep-count').textContent = 'Sweeps: ' + data.sweep_count;
        }
        if (data.total_detections != null) {
            document.getElementById('total-detections').textContent = 'Detections: ' + data.total_detections;
        }
        if (data.last_sweep_time) {
            document.getElementById('last-sweep').textContent = 'Last: ' + data.last_sweep_time;
        }
    }

    function pollStatus() {
        fetch('/api/status')
            .then(function(r) { return r.json(); })
            .then(function(data) { updateStatus(data); })
            .catch(function() {});

        // Poll every 3 seconds
        setTimeout(pollStatus, 3000);
    }

    // --- Event Listeners ---

    document.getElementById('btn-filter').addEventListener('click', loadHistory);
    document.getElementById('btn-refresh').addEventListener('click', loadHistory);

    // Column sorting
    document.querySelectorAll('#history-table th[data-sort]').forEach(function(th) {
        th.addEventListener('click', function() {
            const col = this.dataset.sort;
            if (sortColumn === col) {
                sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                sortColumn = col;
                sortDirection = 'desc';
            }
            loadHistory();
        });
    });

    // --- Utilities ---

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

})();
