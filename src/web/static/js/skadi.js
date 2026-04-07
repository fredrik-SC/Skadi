/* Skadi RF Signal Identification — Dashboard Client */
/* Vanilla JS, no external dependencies */

(function() {
    'use strict';

    var socket = io();
    var sortColumn = 'timestamp_utc';
    var sortDirection = 'desc';
    var selectedId = null;

    // Load data immediately
    loadHistory();
    pollStatus();

    // --- SocketIO Events ---

    socket.on('connect', function() {
        console.log('Connected to Skadi');
        loadHistory();
    });

    socket.on('new_detections', function(data) {
        if (data.detections && data.detections.length > 0) {
            addThreatAlerts(data.detections);
            loadHistory();
        }
    });

    socket.on('scan_status', function(data) {
        updateStatus(data);
    });

    // --- Threat Alerts (HIGH/CRITICAL only) ---

    function addThreatAlerts(detections) {
        var threats = detections.filter(function(d) {
            return d.threat_level === 'CRITICAL' || d.threat_level === 'HIGH';
        });
        if (threats.length === 0) return;

        var feed = document.getElementById('alert-feed');
        var empty = feed.querySelector('.empty-state');
        if (empty) empty.remove();

        threats.forEach(function(det) {
            var card = createAlertCard(det);
            feed.insertBefore(card, feed.firstChild);
        });

        while (feed.children.length > 50) {
            feed.removeChild(feed.lastChild);
        }
    }

    function createAlertCard(det) {
        var threat = det.threat_level || 'HIGH';
        var card = document.createElement('div');
        card.className = 'alert-card threat-' + threat;
        card.style.cursor = 'pointer';
        card.onclick = function() { showDetail(det.id); };

        var freqMhz = (det.frequency_hz / 1e6).toFixed(3);
        var conf = det.confidence_score != null ? det.confidence_score.toFixed(2) : '-';
        var power = det.signal_strength_dbm != null ? det.signal_strength_dbm.toFixed(1) : '-';
        var sigType = det.signal_type || 'UNKNOWN';
        var mod = det.modulation || '?';
        var time = det.timestamp_utc ? det.timestamp_utc.substring(11, 19) + 'Z' : '';

        card.innerHTML =
            '<span class="alert-threat ' + threat + '">' + threat + '</span>' +
            '<span class="alert-type">' + escapeHtml(sigType) + '</span>' +
            '<span class="alert-freq">' + freqMhz + ' MHz</span>' +
            '<span class="alert-mod">' + escapeHtml(mod) + '</span>' +
            '<span class="alert-conf">' + conf + '</span>' +
            '<span class="alert-power">' + power + ' dBm</span>';

        return card;
    }

    // --- History Table ---

    function loadHistory() {
        var params = new URLSearchParams();
        var threatFilter = document.getElementById('filter-threat').value;
        var freqMin = document.getElementById('filter-freq-min').value;
        var freqMax = document.getElementById('filter-freq-max').value;

        if (threatFilter) params.set('threat_level', threatFilter);
        if (freqMin) params.set('freq_min', parseFloat(freqMin) * 1e6);
        if (freqMax) params.set('freq_max', parseFloat(freqMax) * 1e6);
        params.set('limit', '500');

        fetch('/api/detections?' + params.toString())
            .then(function(r) { return r.json(); })
            .then(function(data) { renderHistoryTable(data); })
            .catch(function(err) { console.error('History load error:', err); });
    }

    function renderHistoryTable(rows) {
        rows.sort(function(a, b) {
            var va = a[sortColumn], vb = b[sortColumn];
            if (va == null) va = '';
            if (vb == null) vb = '';
            if (typeof va === 'number' && typeof vb === 'number') {
                return sortDirection === 'asc' ? va - vb : vb - va;
            }
            return sortDirection === 'asc' ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
        });

        var tbody = document.getElementById('history-body');
        tbody.innerHTML = '';

        rows.forEach(function(row) {
            var tr = document.createElement('tr');
            var threat = row.threat_level || 'MEDIUM';
            var freqMhz = row.frequency_hz != null ? (row.frequency_hz / 1e6).toFixed(3) : '-';
            var bwKhz = row.bandwidth_hz != null ? (row.bandwidth_hz / 1e3).toFixed(1) : '-';
            var conf = row.confidence_score != null ? row.confidence_score.toFixed(2) : '-';
            var power = row.signal_strength_dbm != null ? row.signal_strength_dbm.toFixed(1) : '-';
            var time = row.timestamp_utc ? row.timestamp_utc.substring(11, 19) + 'Z' : '-';

            if (row.id === selectedId) tr.className = 'selected';

            tr.innerHTML =
                '<td>' + time + '</td>' +
                '<td style="font-family:monospace;color:var(--accent)">' + freqMhz + '</td>' +
                '<td>' + escapeHtml(row.modulation || '?') + '</td>' +
                '<td>' + escapeHtml(row.signal_type || 'UNKNOWN') + '</td>' +
                '<td>' + conf + '</td>' +
                '<td><span class="threat-badge ' + threat + '">' + threat + '</span></td>' +
                '<td>' + bwKhz + '</td>' +
                '<td>' + power + '</td>';

            tr.onclick = function() { showDetail(row.id); };
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

    // --- Detail Panel ---

    function showDetail(detectionId) {
        if (!detectionId) return;
        selectedId = detectionId;

        fetch('/api/detection/' + detectionId)
            .then(function(r) { return r.json(); })
            .then(function(det) {
                if (det.error) return;
                renderDetailPanel(det);
                document.getElementById('detail-panel').style.display = 'block';
                loadHistory(); // Refresh to highlight selected row
            })
            .catch(function(err) { console.error('Detail load error:', err); });
    }

    function renderDetailPanel(det) {
        var threat = det.threat_level || 'MEDIUM';
        var title = document.getElementById('detail-title');
        title.innerHTML = '<span class="threat-badge ' + threat + '" style="margin-right:8px">' +
            threat + '</span> ' + escapeHtml(det.signal_type || 'UNKNOWN');

        var html = '';

        // Signal info
        html += '<div class="detail-section-title">Signal</div>';
        html += detailRow('Frequency', (det.frequency_hz / 1e6).toFixed(6) + ' MHz');
        html += detailRow('Bandwidth', (det.bandwidth_hz / 1e3).toFixed(1) + ' kHz');
        html += detailRow('Power', det.signal_strength_dbm != null ? det.signal_strength_dbm.toFixed(1) + ' dBm' : '-');
        html += detailRow('Modulation', det.modulation || 'UNKNOWN');
        html += detailRow('Time (UTC)', det.timestamp_utc || '-');

        // Classification
        html += '<div class="detail-section-title">Classification</div>';
        html += detailRow('Signal Type', det.signal_type || 'UNKNOWN');
        html += detailRow('Confidence', det.confidence_score != null ? (det.confidence_score * 100).toFixed(0) + '%' : '-');
        html += detailRow('Alt Match 1', det.alt_match_1 || '-');
        if (det.alt_match_1_confidence != null) {
            html += detailRow('Alt 1 Conf', (det.alt_match_1_confidence * 100).toFixed(0) + '%');
        }
        html += detailRow('Alt Match 2', det.alt_match_2 || '-');
        if (det.alt_match_2_confidence != null) {
            html += detailRow('Alt 2 Conf', (det.alt_match_2_confidence * 100).toFixed(0) + '%');
        }

        // Threat
        html += '<div class="detail-section-title">Assessment</div>';
        html += detailRow('Threat Level', det.threat_level || 'MEDIUM');
        html += detailRow('ACF', det.acf_value != null ? det.acf_value.toFixed(2) + ' ms' : 'None');

        // Known users / description
        if (det.known_users) {
            html += '<div class="detail-section-title">Description</div>';
            html += '<div style="font-size:0.8rem;color:var(--text-secondary);padding:8px 0;line-height:1.5">' +
                escapeHtml(det.known_users) + '</div>';
        }

        document.getElementById('detail-content').innerHTML = html;
    }

    function detailRow(label, value) {
        return '<div class="detail-row"><span class="detail-label">' +
            label + '</span><span class="detail-value">' +
            escapeHtml(String(value)) + '</span></div>';
    }

    document.getElementById('detail-close').addEventListener('click', function() {
        document.getElementById('detail-panel').style.display = 'none';
        selectedId = null;
        loadHistory();
    });

    // --- Status ---

    function updateStatus(data) {
        var badge = document.getElementById('scanner-status');
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

        var startBtn = document.getElementById('btn-start-scan');
        var stopBtn = document.getElementById('btn-stop-scan');
        if (data.scanning) {
            startBtn.disabled = true;
            stopBtn.disabled = false;
        } else {
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }

        var errorBanner = document.getElementById('error-banner');
        if (data.error) {
            errorBanner.textContent = data.error;
            errorBanner.style.display = 'block';
        } else {
            errorBanner.style.display = 'none';
        }
    }

    function pollStatus() {
        fetch('/api/status')
            .then(function(r) { return r.json(); })
            .then(function(data) { updateStatus(data); })
            .catch(function() {});
        setTimeout(pollStatus, 3000);
    }

    // --- Scan Controls ---

    function startScan() {
        var startMhz = document.getElementById('scan-start').value;
        var stopMhz = document.getElementById('scan-stop').value;
        fetch('/api/scan/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                freq_start: parseFloat(startMhz) * 1e6,
                freq_stop: parseFloat(stopMhz) * 1e6
            })
        });
    }

    function stopScan() {
        fetch('/api/scan/stop', {method: 'POST'});
    }

    // --- Event Listeners ---

    document.getElementById('btn-filter').addEventListener('click', loadHistory);
    document.getElementById('btn-start-scan').addEventListener('click', startScan);
    document.getElementById('btn-stop-scan').addEventListener('click', stopScan);

    document.querySelectorAll('#history-table th[data-sort]').forEach(function(th) {
        th.addEventListener('click', function() {
            var col = this.dataset.sort;
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
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

})();
