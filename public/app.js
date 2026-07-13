// Intercept and send console logs/errors to backend server
(function() {
    const originalError = console.error;
    const originalLog = console.log;
    
    console.error = function(...args) {
        originalError.apply(console, args);
        sendVal('ERROR', args.join(' '));
    };
    
    console.log = function(...args) {
        originalLog.apply(console, args);
        sendVal('LOG', args.join(' '));
    };
    
    function sendVal(type, msg) {
        fetch('/api/log', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: type, message: msg })
        }).catch(err => {});
    }
})();

// Global state variables
let currentTab = 'dashboard';
let lastDetectionId = 0;
let lastAlertId = 0;
let hourlyChartInstance = null;
let currentCameraFilter = 'all';

// Pagination state for history
let historyCurrentPage = 0;
const historyPageSize = 15;
let historyTotal = 0;

// Pagination state for watchlist
let watchlistCurrentPage = 0;
const watchlistPageSize = 10;
let watchlistTotal = 0;

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    // 1. Start live clock
    updateClock();
    setInterval(updateClock, 1000);

    // 2. Initialize Chart.js
    initChart();

    // 3. Set default date filters in history tab (start of today to end of today)
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('filter-start-date').value = today;
    document.getElementById('filter-end-date').value = today;

    // 4. Initial data load
    fetchDashboardData();

    // 5. Start auto-refresh interval for dashboard (every 5.0 seconds)
    // loadCamerasList is NOT included here to avoid re-rendering the stream container every 5s
    setInterval(() => {
        if (currentTab === 'dashboard') {
            fetchDashboardData();
            fetchSystemStatus();
        }
    }, 5000);

    // Refresh camera layout every 30 seconds in case cameras are added/removed
    setInterval(() => {
        if (currentTab === 'dashboard') {
            loadCamerasList();
        }
    }, 30000);

    // 6. Fetch initial system status
    fetchSystemStatus();

    // 7. Load cameras list on startup to populate the selector
    loadCamerasList();
});

// Live clock update
function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('zh-TW', { hour12: false });
    document.getElementById('live-time').innerText = timeString;
}

// Switch tabs
function switchTab(tabName) {
    currentTab = tabName;
    
    // Toggle active classes on nav buttons
    document.querySelectorAll('.nav-item').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = Array.from(document.querySelectorAll('.nav-item')).find(btn => 
        btn.getAttribute('onclick').includes(tabName)
    );
    if (activeBtn) activeBtn.classList.add('active');

    // Toggle active classes on sections
    document.querySelectorAll('.tab-section').forEach(sec => {
        sec.classList.remove('active');
    });
    document.getElementById(`tab-${tabName}`).classList.add('active');

    // Tab-specific loads
    if (tabName === 'history') {
        historyCurrentPage = 0;
        queryHistory();
    } else if (tabName === 'watchlist') {
        loadWatchlist();
    } else if (tabName === 'settings') {
        loadSystemSettings();
        switchSettingsSubTab('camera');
    } else if (tabName === 'zones') {
        zoneEditorInit();
    } else {
        fetchDashboardData();
    }
}

// Switch Settings Sub-tabs
function switchSettingsSubTab(subTabName) {
    // 1. Toggle active classes on sub-nav items
    document.querySelectorAll('.sub-nav-item').forEach(btn => {
        btn.classList.remove('active');
        btn.style.background = 'transparent';
        btn.style.border = '1px solid transparent';
        btn.style.color = 'var(--text-secondary)';
    });

    const activeBtn = Array.from(document.querySelectorAll('.sub-nav-item')).find(btn => 
        btn.getAttribute('onclick').includes(subTabName)
    );
    if (activeBtn) {
        activeBtn.classList.add('active');
        activeBtn.style.background = 'rgba(255, 255, 255, 0.06)';
        activeBtn.style.border = '1px solid rgba(255, 255, 255, 0.1)';
        activeBtn.style.color = 'var(--text-primary)';
    }

    // 2. Toggle active sub-sections
    document.querySelectorAll('.settings-sub-section').forEach(sec => {
        sec.style.display = 'none';
    });
    
    const targetSec = document.getElementById(`settings-sub-${subTabName}`);
    if (targetSec) {
        targetSec.style.display = 'block';
    }
}

// Initialize traffic chart
function initChart() {
    const ctx = document.getElementById('hourlyChart').getContext('2d');
    
    const carGradient = ctx.createLinearGradient(0, 0, 0, 200);
    carGradient.addColorStop(0, 'rgba(59, 130, 246, 0.4)');
    carGradient.addColorStop(1, 'rgba(59, 130, 246, 0.02)');

    const motoGradient = ctx.createLinearGradient(0, 0, 0, 200);
    motoGradient.addColorStop(0, 'rgba(16, 185, 129, 0.4)');
    motoGradient.addColorStop(1, 'rgba(16, 185, 129, 0.02)');

    const hours = Array.from({ length: 24 }, (_, i) => `${i.toString().padStart(2, '0')}:00`);

    hourlyChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: hours,
            datasets: [
                {
                    label: '汽車',
                    data: Array(24).fill(0),
                    backgroundColor: carGradient,
                    borderColor: '#3B82F6',
                    borderWidth: 2,
                    borderRadius: 4,
                    barPercentage: 0.5,
                    categoryPercentage: 0.8
                },
                {
                    label: '機車',
                    data: Array(24).fill(0),
                    backgroundColor: motoGradient,
                    borderColor: '#10B981',
                    borderWidth: 2,
                    borderRadius: 4,
                    barPercentage: 0.5,
                    categoryPercentage: 0.8
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            // Disable all animations to prevent chart flash/flicker on data update
            animation: false,
            transitions: {
                active: { animation: { duration: 0 } }
            },
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#9CA3AF',
                        font: { family: 'Outfit, Noto Sans TC', size: 11 }
                    }
                },
                tooltip: {
                    backgroundColor: '#111827',
                    titleColor: '#F3F4F6',
                    bodyColor: '#10B981',
                    borderColor: 'rgba(255, 255, 255, 0.08)',
                    borderWidth: 1,
                    font: {
                        family: 'Outfit, Noto Sans TC'
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#9CA3AF',
                        font: { size: 10, family: 'Outfit' }
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.04)'
                    },
                    ticks: {
                        color: '#9CA3AF',
                        font: { size: 10, family: 'Outfit' },
                        stepSize: 1
                    }
                }
            }
        }
    });
}

let consecutiveFailures = 0;
const FAILURE_THRESHOLD = 10;  // Increased from 3 to prevent flicker on minor API blips

function handleFetchFailure(err, context) {
    console.error(`${context} error:`, err);
    consecutiveFailures++;
    if (consecutiveFailures >= FAILURE_THRESHOLD && isServerOnline) {
        isServerOnline = false;
        showStreamFallback(true);
    }
}

function handleFetchSuccess() {
    consecutiveFailures = 0;
    if (!isServerOnline) {
        isServerOnline = true;
        showStreamFallback(false);
        // Do NOT call reloadStreams() here - MJPEG streams are persistent long connections.
        // Resetting img.src causes a black frame flash. The browser auto-reconnects MJPEG.
    }
}

// Fetch dashboard data (recent list, stats & alerts)
async function fetchDashboardData() {
    try {
        const cameraId = currentCameraFilter;
        
        let detectionsUrl = '/api/detections?limit=15';
        let statsUrl = '/api/stats';
        let alertsUrl = '/api/alerts';
        
        if (cameraId && cameraId !== 'all') {
            detectionsUrl += `&camera_id=${cameraId}`;
            statsUrl += `?camera_id=${cameraId}`;
            alertsUrl += `?camera_id=${cameraId}`;
        }

        // A. Fetch recent detections (limit 15 for sidebar)
        let detectionsRes;
        try {
            detectionsRes = await fetch(detectionsUrl);
            if (!detectionsRes.ok) throw new Error("Network error: " + detectionsRes.status);
            const text = await detectionsRes.text();
            if (!text) throw new Error("Empty response");
            const payload = JSON.parse(text);
            updateLiveFeed(payload.data);
            handleFetchSuccess();
        } catch (e) {
            console.error("Detections fetch failed:", detectionsUrl, e);
            throw e;
        }

        // B. Fetch stats
        try {
            const statsRes = await fetch(statsUrl);
            if (statsRes.ok) {
                const text = await statsRes.text();
                if (text) {
                    const stats = JSON.parse(text);
                    updateStatsUI(stats);
                }
            }
        } catch (e) {
            console.error("Stats fetch failed:", statsUrl, e);
        }

        // C. Fetch watchlist alerts for today
        try {
            const alertsRes = await fetch(alertsUrl);
            if (alertsRes.ok) {
                const text = await alertsRes.text();
                if (text) {
                    const alerts = JSON.parse(text);
                    updateAlertsUI(alerts);
                }
            }
        } catch (e) {
            console.error("Alerts fetch failed:", alertsUrl, e);
        }

    } catch (err) {
        handleFetchFailure(err, "Dashboard fetch");
    }
}

let lastFeedLatestId = 0;
let lastRenderedDetectionIds = new Set();

// Update live sidebar feed list - incremental update to prevent img reload flicker
function updateLiveFeed(detections) {
    const badge = document.getElementById('feed-count-badge');
    const feedList = document.getElementById('live-feed-list');

    if (!feedList) return;

    if (detections.length === 0) {
        if (badge) badge.innerText = 0;
        if (lastFeedLatestId !== -1) {
            feedList.innerHTML = `
                <div class="feed-empty">
                    <i class="fa-solid fa-circle-notch fa-spin"></i>
                    <p>正在等待車輛經過...</p>
                </div>
            `;
            lastRenderedDetectionIds.clear();
            lastFeedLatestId = -1;
        }
        return;
    }

    const latestId = detections[0].id;
    if (lastFeedLatestId === latestId) {
        return; // No new detections - do NOT touch DOM at all, prevents all flicker!
    }

    if (badge) badge.innerText = detections.length;

    // Check if there is a new detection to trigger animation
    const latest = detections[0];
    const isNew = lastDetectionId !== 0 && latest.id > lastDetectionId;
    lastDetectionId = latest.id;
    lastFeedLatestId = latestId;

    // Update latest plate stat box dynamically
    const latestPlateEl = document.getElementById('stat-latest');
    if (latestPlateEl) {
        const iconHtml = latest.vehicle_type === 'MOTORCYCLE'
            ? '<i class="fa-solid fa-motorcycle" style="margin-right: 6px; font-size: 14px; opacity: 0.8;"></i>'
            : '<i class="fa-solid fa-car" style="margin-right: 6px; font-size: 14px; opacity: 0.8;"></i>';
        latestPlateEl.innerHTML = `${iconHtml}${latest.plate_number}`;
    }

    // Find which detection IDs are new (not yet rendered)
    const newDetections = detections.filter(d => !lastRenderedDetectionIds.has(d.id));

    if (newDetections.length === 0) {
        // No truly new items, just trim excess items if list grew
        const existingItems = feedList.querySelectorAll('.feed-item');
        while (existingItems.length > detections.length && feedList.lastChild) {
            feedList.removeChild(feedList.lastChild);
        }
        return;
    }

    // Clear the "waiting" empty state if present
    const emptyEl = feedList.querySelector('.feed-empty');
    if (emptyEl) feedList.innerHTML = '';

    // Prepend only the genuinely new items (no re-rendering of existing ones = no img flicker)
    newDetections.reverse().forEach((det, idx) => {
        lastRenderedDetectionIds.add(det.id);
        const highlightClass = (det.id === latestId && isNew) ? 'highlight-new' : '';
        const watchLabel = det.watch_category ? ` <span class="watch-badge ${det.watch_category.toLowerCase()}" style="font-size:9px; padding: 2px 4px;">${det.watch_category}</span>` : '';
        const vIcon = det.vehicle_type === 'MOTORCYCLE'
            ? '<i class="fa-solid fa-motorcycle" style="margin-right: 6px; font-size: 13px; opacity: 0.7;"></i>'
            : '<i class="fa-solid fa-car" style="margin-right: 6px; font-size: 13px; opacity: 0.7;"></i>';
        const camLabel = det.camera_name ? `<span style="color:var(--color-primary); margin-left: 8px; font-weight: 500;">[${det.camera_name}]</span>` : '';

        const div = document.createElement('div');
        div.className = `feed-item ${highlightClass}`;
        div.setAttribute('onclick', `openModal('${det.plate_number}', '${det.local_time}', '${det.full_image_path}')`);
        div.dataset.detId = det.id;
        div.innerHTML = `
            <img class="feed-item-img" src="/crops/${det.crop_image_path}" alt="Plate crop" onerror="this.src='https://placehold.co/90x48/111827/ffffff?text=Crop'">
            <div class="feed-item-info">
                <span class="feed-item-plate">${vIcon}${det.plate_number}${watchLabel}</span>
                <div class="feed-item-meta">
                    <span><i class="fa-regular fa-clock"></i> ${det.local_time.split(' ')[1]}${camLabel}</span>
                    <span class="feed-item-conf">信心度: ${Math.round(det.confidence * 100)}%</span>
                </div>
            </div>
        `;
        feedList.insertBefore(div, feedList.firstChild);
    });

    // Remove items beyond the max display count to keep the list tidy
    const maxItems = detections.length;
    const allItems = feedList.querySelectorAll('.feed-item');
    for (let i = maxItems; i < allItems.length; i++) {
        const removedId = parseInt(allItems[i].dataset.detId);
        lastRenderedDetectionIds.delete(removedId);
        feedList.removeChild(allItems[i]);
    }
}



let lastStatsJson = "";

// Update dashboard stats cards
function updateStatsUI(stats) {
    const statsJson = JSON.stringify(stats);
    if (lastStatsJson === statsJson) {
        return;
    }
    lastStatsJson = statsJson;

    document.getElementById('stat-total').innerText = stats.total_today;
    
    // Update split stats card
    const splitEl = document.getElementById('stat-split');
    if (splitEl) {
        splitEl.innerText = `${stats.cars_today || 0} / ${stats.motorcycles_today || 0}`;
    }

    // Update Chart - use 'none' mode to skip animation and prevent chart flash
    if (hourlyChartInstance) {
        if (stats.hourly_cars_today && stats.hourly_motorcycles_today) {
            hourlyChartInstance.data.datasets[0].data = stats.hourly_cars_today;
            hourlyChartInstance.data.datasets[1].data = stats.hourly_motorcycles_today;
            hourlyChartInstance.update('none');
        } else if (stats.hourly_today) {
            hourlyChartInstance.data.datasets[0].data = stats.hourly_today;
            hourlyChartInstance.update('none');
        }
    }
}

let lastAlertsJson = "";

// Update Watchlist Alerts UI
function updateAlertsUI(alerts) {
    const alertsJson = JSON.stringify(alerts);
    if (lastAlertsJson === alertsJson) {
        return;
    }
    lastAlertsJson = alertsJson;

    const board = document.getElementById('today-alerts-board');
    const badge = document.getElementById('alerts-count-badge');
    const list = document.getElementById('today-alerts-list');

    badge.innerText = alerts.length;

    if (alerts.length === 0) {
        board.classList.add('hidden');
        return;
    }

    // Show board
    board.classList.remove('hidden');

    // Check for a new alert to trigger audio and banner notification
    const latest = alerts[0];
    if (lastAlertId !== 0 && latest.id > lastAlertId) {
        triggerAlertNotification(latest);
    }
    lastAlertId = latest.id;

    // Render list
    let listHtml = '';
    alerts.forEach(alert => {
        const vIcon = alert.vehicle_type === 'MOTORCYCLE' 
            ? '<i class="fa-solid fa-motorcycle" style="margin-right: 6px; font-size: 12px; opacity: 0.7;"></i>' 
            : '<i class="fa-solid fa-car" style="margin-right: 6px; font-size: 12px; opacity: 0.7;"></i>';
        const camBadge = alert.camera_name ? ` <span class="watch-badge suspicious" style="background-color: rgba(59, 130, 246, 0.15); color: var(--color-blue); border: 1px solid rgba(59, 130, 246, 0.3); font-size:9px; padding: 2px 4px;">${alert.camera_name}</span>` : '';
        const actionClass = alert.action === 'EXIT' ? 'exit' : 'entry';
        const actionText = alert.action === 'EXIT' ? '離開' : '進入';
        const actionBadge = `<span class="action-badge ${actionClass}">${actionText}</span>`;

        listHtml += `
            <div class="alert-summary-item">
                <div class="alert-item-left">
                    <span class="alert-item-time"><i class="fa-regular fa-clock"></i> ${alert.local_time.split(' ')[1]}</span>
                    <span class="alert-item-plate">${vIcon}${alert.plate_number}</span>
                    <span class="watch-badge ${alert.category.toLowerCase()}">${alert.category}</span>${actionBadge}${camBadge}
                </div>
                <span class="alert-item-desc">${alert.description || '無備註'}</span>
            </div>
        `;
    });
    list.innerHTML = listHtml;
}

// Trigger alert banner and buzzer
function triggerAlertNotification(alert) {
    // 1. Play alert sound
    const audio = document.getElementById('alert-sound');
    if (audio) {
        audio.currentTime = 0;
        audio.play().catch(err => console.log("Audio play blocked by browser policy"));
    }

    // 2. Display banner
    const actionText = alert.action === 'EXIT' ? '離開' : '進入';
    const banner = document.getElementById('alert-banner');
    const bannerText = document.getElementById('alert-banner-text');
    bannerText.innerText = `[${alert.category}] 車牌號碼 ${alert.plate_number} (${actionText}) 剛剛由【${alert.camera_name || '未知'}】通過！(備註: ${alert.description || '無'})`;
    
    banner.classList.remove('hidden');

    // Auto-hide alert banner after 6 seconds
    setTimeout(closeAlertBanner, 6000);
}

function closeAlertBanner() {
    document.getElementById('alert-banner').classList.add('hidden');
}

function toggleAlertsBoard() {
    const board = document.getElementById('today-alerts-board');
    if (board) {
        board.classList.toggle('collapsed');
    }
}

// Query History list (with Pagination & Range Filters)
async function queryHistory(page = 0) {
    historyCurrentPage = page;
    const searchVal = document.getElementById('filter-search').value.trim();
    const startDate = document.getElementById('filter-start-date').value;
    const endDate = document.getElementById('filter-end-date').value;

    const tbody = document.getElementById('history-tbody');
    const emptyDiv = document.getElementById('history-empty');

    tbody.innerHTML = `
        <tr>
            <td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 40px 0;">
                <i class="fa-solid fa-spinner fa-spin"></i> 正在查詢資料...
            </td>
        </tr>
    `;

    try {
        const offset = historyCurrentPage * historyPageSize;
        let url = `/api/detections?limit=${historyPageSize}&offset=${offset}`;
        if (currentCameraFilter && currentCameraFilter !== 'all') {
            url += `&camera_id=${currentCameraFilter}`;
        }
        if (searchVal) url += `&search=${encodeURIComponent(searchVal)}`;
        if (startDate) url += `&start_date=${encodeURIComponent(startDate)}`;
        if (endDate) url += `&end_date=${encodeURIComponent(endDate)}`;

        const res = await fetch(url);
        if (!res.ok) throw new Error("Fetch failed");
        const payload = await res.json();

        const total = payload.total;
        const data = payload.data;
        historyTotal = total;

        if (data.length === 0) {
            tbody.innerHTML = '';
            emptyDiv.classList.remove('hidden');
            updatePaginationUI(0);
            return;
        }

        emptyDiv.classList.add('hidden');
        let tableRows = '';
        data.forEach(det => {
            let watchTag = '-';
            if (det.watch_category) {
                watchTag = `<span class="watch-badge ${det.watch_category.toLowerCase()}">${det.watch_category}</span>`;
            }
            const vIcon = det.vehicle_type === 'MOTORCYCLE' 
                ? '<i class="fa-solid fa-motorcycle" style="margin-right: 6px; font-size: 12px; opacity: 0.7;"></i>' 
                : '<i class="fa-solid fa-car" style="margin-right: 6px; font-size: 12px; opacity: 0.7;"></i>';
            
            tableRows += `
                <tr>
                    <td>
                        ${det.local_time}
                        <div style="font-size: 11px; color: var(--color-primary); opacity: 0.85; margin-top: 3px; font-weight: 500;">
                            <i class="fa-solid fa-video" style="font-size: 9px; margin-right: 4px;"></i>${det.camera_name || '未知'}
                        </div>
                    </td>
                    <td>
                        <div style="display: inline-flex; align-items: center; gap: 8px;">
                            ${vIcon}<span class="table-plate-text">${det.plate_number}</span>
                            <button class="btn-edit-plate" onclick="openEditModal(${det.id}, '${det.plate_number}')" title="修正車牌">
                                <i class="fa-solid fa-pen"></i>
                            </button>
                        </div>
                    </td>
                    <td><span class="table-conf-text">${Math.round(det.confidence * 100)}%</span></td>
                    <td>${watchTag}</td>
                    <td>
                        <img class="table-img" src="/crops/${det.crop_image_path}" alt="Plate crop" 
                             onclick="openModal('${det.plate_number}', '${det.local_time}', '${det.crop_image_path}', true)">
                    </td>
                    <td>
                        <img class="table-img" src="/fulls/${det.full_image_path}" alt="Full capture"
                             onclick="openModal('${det.plate_number}', '${det.local_time}', '${det.full_image_path}', false)">
                    </td>
                </tr>
            `;
        });
        tbody.innerHTML = tableRows;
        
        // Update pagination page buttons
        updatePaginationUI(total);

    } catch (err) {
        tbody.innerHTML = `
            <tr>
                <td colspan="6" style="text-align: center; color: var(--color-red); padding: 40px 0;">
                    <i class="fa-solid fa-triangle-exclamation"></i> 查詢出錯，請確認後端連線。
                </td>
            </tr>
        `;
        console.error("Query history error:", err);
    }
}

// Update Pagination button enabling
function updatePaginationUI(total) {
    const indicator = document.getElementById('page-indicator');
    const prevBtn = document.getElementById('btn-prev-page');
    const nextBtn = document.getElementById('btn-next-page');

    const totalPages = Math.ceil(total / historyPageSize) || 1;
    indicator.innerText = `第 ${historyCurrentPage + 1} 頁 / 共 ${totalPages} 頁`;

    prevBtn.disabled = (historyCurrentPage === 0);
    nextBtn.disabled = ((historyCurrentPage + 1) * historyPageSize >= total);
}

// Change history page
function changePage(direction) {
    const nextPage = historyCurrentPage + direction;
    const totalPages = Math.ceil(historyTotal / historyPageSize);
    if (nextPage >= 0 && nextPage < totalPages) {
        queryHistory(nextPage);
    }
}

// Reset history filters
function resetFilters() {
    document.getElementById('filter-search').value = '';
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('filter-start-date').value = today;
    document.getElementById('filter-end-date').value = today;
    queryHistory(0);
}

// Export queries to CSV
async function exportToCSV() {
    const searchVal = document.getElementById('filter-search').value.trim();
    const startDate = document.getElementById('filter-start-date').value;
    const endDate = document.getElementById('filter-end-date').value;

    try {
        // Fetch up to 5000 matches for the export
        let url = `/api/detections?limit=5000`;
        if (currentCameraFilter && currentCameraFilter !== 'all') {
            url += `&camera_id=${currentCameraFilter}`;
        }
        if (searchVal) url += `&search=${encodeURIComponent(searchVal)}`;
        if (startDate) url += `&start_date=${encodeURIComponent(startDate)}`;
        if (endDate) url += `&end_date=${encodeURIComponent(endDate)}`;

        const res = await fetch(url);
        if (!res.ok) throw new Error("CSV Fetch failed");
        const payload = await res.json();
        const data = payload.data;

        if (data.length === 0) {
            alert("目前條件下沒有任何紀錄可供匯出。");
            return;
        }

        // CSV Header
        let csvContent = "時間,車牌號碼,辨識信心度,追蹤類別,備註\n";
        
        data.forEach(row => {
            const time = row.local_time;
            const plate = row.plate_number;
            const conf = `${Math.round(row.confidence * 100)}%`;
            const watchCat = row.watch_category || "無";
            const watchDesc = (row.watch_description || "無").replace(/,/g, "，"); // Replace commas to avoid splitting CSV columns
            
            csvContent += `${time},${plate},${conf},${watchCat},${watchDesc}\n`;
        });

        // Add BOM (Byte Order Mark) to ensure Excel opens Traditional Chinese characters correctly
        const blob = new Blob([new Uint8Array([0xEF, 0xBB, 0xBF]), csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement("a");
        const urlBlob = URL.createObjectURL(blob);
        
        link.setAttribute("href", urlBlob);
        link.setAttribute("download", `車牌進出紀錄匯出_${startDate}_${endDate}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

    } catch (err) {
        alert("匯出失敗，請確認連線。");
        console.error("Export CSV error:", err);
    }
}

// ==========================================================================
// Watchlist Management functions (Newly Added)
// ==========================================================================

// Load Watchlist from Server
async function loadWatchlist(page = 0) {
    watchlistCurrentPage = page;
    const tbody = document.getElementById('watchlist-tbody');
    const emptyDiv = document.getElementById('watchlist-empty');

    tbody.innerHTML = `
        <tr>
            <td colspan="5" style="text-align: center; color: var(--text-secondary); padding: 20px 0;">
                <i class="fa-solid fa-spinner fa-spin"></i> 正在載入名單...
            </td>
        </tr>
    `;

    try {
        const offset = watchlistCurrentPage * watchlistPageSize;
        const res = await fetch(`/api/watchlist?limit=${watchlistPageSize}&offset=${offset}`);
        if (!res.ok) throw new Error("Load failed");
        const payload = await res.json();

        const total = payload.total;
        const list = payload.data;
        watchlistTotal = total;

        if (list.length === 0) {
            tbody.innerHTML = '';
            emptyDiv.classList.remove('hidden');
            updateWatchlistPaginationUI(0);
            return;
        }

        emptyDiv.classList.add('hidden');
        let rowsHtml = '';
        list.forEach(item => {
            rowsHtml += `
                <tr>
                    <td><span class="table-plate-text">${item.plate_number}</span></td>
                    <td><span class="watch-badge ${item.category.toLowerCase()}">${item.category}</span></td>
                    <td>${item.description || '-'}</td>
                    <td>${item.created_at}</td>
                    <td>
                        <button class="btn-delete" onclick="deleteWatchlistItem('${item.plate_number}')">
                            <i class="fa-solid fa-trash-can"></i> 刪除
                        </button>
                    </td>
                </tr>
            `;
        });
        tbody.innerHTML = rowsHtml;

        updateWatchlistPaginationUI(total);

    } catch (err) {
        tbody.innerHTML = `
            <tr>
                <td colspan="5" style="text-align: center; color: var(--color-red); padding: 20px 0;">
                    <i class="fa-solid fa-triangle-exclamation"></i> 載入名單失敗。
                </td>
            </tr>
        `;
        console.error("Load watchlist error:", err);
    }
}

// Update Watchlist Pagination button enabling
function updateWatchlistPaginationUI(total) {
    const indicator = document.getElementById('watchlist-page-indicator');
    const prevBtn = document.getElementById('btn-watchlist-prev-page');
    const nextBtn = document.getElementById('btn-watchlist-next-page');

    const totalPages = Math.ceil(total / watchlistPageSize) || 1;
    indicator.innerText = `第 ${watchlistCurrentPage + 1} 頁 / 共 ${totalPages} 頁`;

    prevBtn.disabled = (watchlistCurrentPage === 0);
    nextBtn.disabled = ((watchlistCurrentPage + 1) * watchlistPageSize >= total);
}

// Change watchlist page
function changeWatchlistPage(direction) {
    const nextPage = watchlistCurrentPage + direction;
    const totalPages = Math.ceil(watchlistTotal / watchlistPageSize);
    if (nextPage >= 0 && nextPage < totalPages) {
        loadWatchlist(nextPage);
    }
}

// Add a plate to Watchlist
async function addWatchlistItem(event) {
    event.preventDefault();
    
    const plateInput = document.getElementById('watch-plate');
    const categorySelect = document.getElementById('watch-category');
    const descInput = document.getElementById('watch-description');

    const plate = plateInput.value.trim().toUpperCase();
    const category = categorySelect.value;
    const desc = descInput.value.trim();

    if (!plate) return;

    try {
        const res = await fetch('/api/watchlist', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                plate_number: plate,
                category: category,
                description: desc
            })
        });

        if (!res.ok) throw new Error("POST failed");
        const result = await res.json();

        if (result.success) {
            // Reset input values
            plateInput.value = '';
            descInput.value = '';
            // Reload table list, go to page 0 since newly added item is at the top
            loadWatchlist(0);
        } else {
            alert("儲存失敗：" + result.error);
        }

    } catch (err) {
        alert("連線失敗，請檢查後端服務。");
        console.error("Add watchlist error:", err);
    }
}

// Delete item from Watchlist
async function deleteWatchlistItem(plate) {
    if (!confirm(`確定要將車牌 ${plate} 移出追蹤名單嗎？`)) {
        return;
    }

    try {
        const res = await fetch(`/api/watchlist?plate_number=${encodeURIComponent(plate)}`, {
            method: 'DELETE'
        });

        if (!res.ok) throw new Error("DELETE failed");
        const result = await res.json();

        if (result.success) {
            // Handle page shifting if the deleted item was the last one on the current page
            const totalAfterDelete = watchlistTotal - 1;
            const totalPagesAfterDelete = Math.ceil(totalAfterDelete / watchlistPageSize) || 1;
            let targetPage = watchlistCurrentPage;
            if (targetPage >= totalPagesAfterDelete) {
                targetPage = Math.max(0, totalPagesAfterDelete - 1);
            }
            loadWatchlist(targetPage);
        } else {
            alert("刪除失敗：" + result.error);
        }

    } catch (err) {
        alert("刪除連線失敗，請檢查網路。");
        console.error("Delete watchlist error:", err);
    }
}


// === Canvas Polling: replaces MJPEG <img> to eliminate white-flash/reconnect flicker ===
// Each camera gets an independent JS poller that fetches /api/frame every 100ms.
// No persistent connection = no disconnect = no blank frame flicker.
const streamPollers = {};  // cam_id -> intervalId

function startStreamPoller(camId) {
    stopStreamPoller(camId);  // cancel any existing poller for this cam
    const canvas = document.getElementById(`stream-canvas-${camId}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = 960;
    canvas.height = 540;
    let loading = false;

    const poll = async () => {
        if (loading) return;
        loading = true;
        try {
            const res = await fetch(`/api/frame?id=${camId}`, { cache: 'no-store' });
            if (res.ok) {
                const blob = await res.blob();
                const url = URL.createObjectURL(blob);
                const img = new Image();
                img.onload = () => { ctx.drawImage(img, 0, 0, 960, 540); URL.revokeObjectURL(url); loading = false; };
                img.onerror = () => { URL.revokeObjectURL(url); loading = false; };
                img.src = url;
            } else {
                loading = false;  // 204 No Content: engine not ready yet
            }
        } catch(e) { loading = false; }
    };
    poll();  // immediate first frame
    streamPollers[camId] = setInterval(poll, 200);  // 5fps (reduced from 10fps to cut HTTP load)
}

function stopStreamPoller(camId) {
    if (streamPollers[camId] != null) {
        clearInterval(streamPollers[camId]);
        delete streamPollers[camId];
    }
}

function stopAllStreamPollers() {
    Object.keys(streamPollers).forEach(stopStreamPoller);
}

let activeCameraIdsCache = "";

function updateVideoStreamLayout(cameras) {
    let activeCams = cameras.filter(c => c.is_active === 1);
    
    if (currentCameraFilter !== 'all') {
        const filterId = parseInt(currentCameraFilter);
        activeCams = activeCams.filter(c => c.id === filterId);
    }
    
    const cacheKey = currentCameraFilter + "_" + activeCams.map(c => c.id).sort((a, b) => a - b).join(",");
    if (activeCameraIdsCache === cacheKey) {
        return;  // No layout change, existing pollers keep running
    }
    
    activeCameraIdsCache = cacheKey;
    stopAllStreamPollers();  // Stop old pollers before rebuilding DOM
    
    const container = document.querySelector('.video-container');
    if (!container) return;
    
    // Canvas style: fills container, GPU-isolated to prevent pulse animation repaints
    const canvasStyle = 'display:block; width:100%; height:100%; object-fit:contain; contain:strict; isolation:isolate;';
    
    if (activeCams.length > 1) {
        container.classList.add('split-layout');
        let html = '';
        activeCams.forEach(cam => {
            html += `
                <div class="stream-item" id="stream-item-${cam.id}">
                    <span class="stream-title">
                        <i class="fa-solid fa-video"></i> ${cam.name}
                    </span>
                    <canvas id="stream-canvas-${cam.id}" style="${canvasStyle}"></canvas>
                    <div id="stream-fallback-${cam.id}" class="stream-fallback hidden">
                        <i class="fa-solid fa-triangle-exclamation" style="color:var(--color-yellow); font-size: 24px;"></i>
                        <p style="font-size: 11px; margin-top: 8px;">無法取得影像串流，請確認後端服務是否正常執行</p>
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;
        activeCams.forEach(cam => startStreamPoller(cam.id));
    } else {
        container.classList.remove('split-layout');
        const activeCam = activeCams[0] || cameras[0];
        const camId = activeCam ? activeCam.id : '';
        
        container.innerHTML = `
            <canvas id="stream-canvas-${camId}" style="${canvasStyle}"></canvas>
            <div id="stream-fallback" class="stream-fallback hidden">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <p>無法取得影像串流，請確認後端服務是否正常執行</p>
            </div>
        `;
        if (camId) startStreamPoller(camId);
    }
}

let isServerOnline = true;

function showStreamFallback(show) {
    // Use CSS opacity overlay instead of hiding the img element.
    // This prevents the black flash caused by removing src or hidden/show of the img.
    const headerSelect = document.getElementById('header-camera-select');
    const currentCamFilter = headerSelect ? headerSelect.value : 'all';
    
    if (currentCamFilter === 'all') {
        const streamItems = document.querySelectorAll('.stream-item');
        streamItems.forEach(item => {
            const camId = item.id.split('-').pop();
            const img = document.getElementById(`stream-img-${camId}`);
            const fallback = document.getElementById(`stream-fallback-${camId}`);
            if (show) {
                // Dim image but keep it visible to avoid black flash
                if (img) img.style.opacity = '0.25';
                if (fallback) fallback.classList.remove('hidden');
            } else {
                if (img) img.style.opacity = '1';
                if (fallback) fallback.classList.add('hidden');
            }
        });
    } else {
        const img = document.getElementById('stream-img');
        const fallback = document.getElementById('stream-fallback');
        if (show) {
            if (img) img.style.opacity = '0.25';
            if (fallback) fallback.classList.remove('hidden');
        } else {
            if (img) img.style.opacity = '1';
            if (fallback) fallback.classList.add('hidden');
        }
    }
}

function reloadStreams() {
    const headerSelect = document.getElementById('header-camera-select');
    const currentCamFilter = headerSelect ? headerSelect.value : 'all';
    if (currentCamFilter === 'all') {
        const streamItems = document.querySelectorAll('.stream-item');
        streamItems.forEach(item => {
            const camId = item.id.split('-').pop();
            const img = document.getElementById(`stream-img-${camId}`);
            if (img) {
                img.src = `/api/stream?id=${camId}&t=${Date.now()}`;
            }
        });
    } else {
        const img = document.getElementById('stream-img');
        if (img) {
            img.src = `/api/stream?id=${currentCamFilter}&t=${Date.now()}`;
        }
    }
}

// Modal popups for pictures
function openModal(plate, time, imagePath, isCrop = false) {
    const modal = document.getElementById('image-modal');
    const modalImg = document.getElementById('modal-img');
    const modalPlate = document.getElementById('modal-plate');
    const modalTime = document.getElementById('modal-time');

    modalPlate.innerText = plate;
    modalTime.innerText = time;
    
    // Set appropriate image source
    const subPath = isCrop ? '/crops/' : '/fulls/';
    modalImg.src = subPath + imagePath;

    modal.classList.remove('hidden');
}

function closeModal() {
    document.getElementById('image-modal').classList.add('hidden');
}

// Close modals when clicking outside content
window.onclick = function(event) {
    const modal = document.getElementById('image-modal');
    const editModal = document.getElementById('edit-modal');
    if (event.target === modal) {
        closeModal();
    }
    if (event.target === editModal) {
        closeEditModal();
    }
}

// System status control
let systemStatus = 'running';

async function fetchSystemStatus() {
    try {
        const res = await fetch('/api/system_status');
        if (res.ok) {
            const data = await res.json();
            updateSystemStatusUI(data.status, data.gate);
            handleFetchSuccess();
        } else {
            throw new Error("Server returned non-OK status");
        }
    } catch (err) {
        handleFetchFailure(err, "Fetch system status");
    }
}

async function toggleSystemStatus() {
    const nextStatus = systemStatus === 'running' ? 'paused' : 'running';
    try {
        const res = await fetch('/api/system_status', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ status: nextStatus })
        });
        if (res.ok) {
            const data = await res.json();
            updateSystemStatusUI(data.status, data.gate);
        }
    } catch (err) {
        alert("無法連接後端服務，操作失敗。");
    }
}

let lastSystemStatus = null;
let lastGateStatus = null;

function updateSystemStatusUI(status, gateStatus) {
    // Dedup: only update DOM if values actually changed
    const statusChanged = status !== lastSystemStatus;
    const gateChanged = gateStatus !== lastGateStatus;
    if (!statusChanged && !gateChanged) return;

    systemStatus = status;

    if (statusChanged) {
        lastSystemStatus = status;
        const btn = document.getElementById('btn-system-toggle');
        const pulse = document.getElementById('status-pulse');
        const text = document.getElementById('status-text');
        if (status === 'running') {
            if (btn.dataset.state !== 'running') {
                btn.innerHTML = `<i class="fa-solid fa-pause"></i> 暫停監控系統`;
                btn.className = "btn btn-primary btn-sm";
                btn.dataset.state = 'running';
            }
            pulse.style.backgroundColor = "var(--color-primary)";
            pulse.style.animation = "pulse 1.6s infinite";
            text.innerText = "相機連接正常";
        } else {
            if (btn.dataset.state !== 'paused') {
                btn.innerHTML = `<i class="fa-solid fa-play"></i> 啟動監控系統`;
                btn.className = "btn btn-secondary btn-sm";
                btn.dataset.state = 'paused';
            }
            pulse.style.backgroundColor = "var(--color-red)";
            pulse.style.animation = "none";
            text.innerText = "監控已暫停";
        }
    }

    if (gateChanged && gateStatus) {
        lastGateStatus = gateStatus;
        const gatePulse = document.getElementById('gate-pulse');
        const gateText = document.getElementById('gate-text');
        if (gatePulse && gateText) {
            if (gateStatus === 'open') {
                gatePulse.style.backgroundColor = "var(--color-primary)";
                gatePulse.style.animation = "pulse 1.6s infinite";
                gateText.innerText = "大門已開啟";
            } else {
                gatePulse.style.backgroundColor = "var(--color-red)";
                gatePulse.style.animation = "none";
                gateText.innerText = "大門已關閉";
            }
        }
    }
}

// ==========================================================================
// Plate Edit / Correction functions
// ==========================================================================

function openEditModal(detId, plateNumber) {
    document.getElementById('edit-det-id').value = detId;
    document.getElementById('edit-plate-input').value = plateNumber;
    document.getElementById('edit-modal').classList.remove('hidden');
    // Focus the input and select text
    setTimeout(() => {
        const input = document.getElementById('edit-plate-input');
        input.focus();
        input.select();
    }, 100);
}

function closeEditModal() {
    document.getElementById('edit-modal').classList.add('hidden');
}

async function submitPlateCorrection(event) {
    event.preventDefault();
    const detId = document.getElementById('edit-det-id').value;
    const newPlate = document.getElementById('edit-plate-input').value.trim().toUpperCase();

    if (!newPlate) return;

    try {
        const res = await fetch('/api/detections/edit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                id: parseInt(detId),
                plate_number: newPlate
            })
        });

        if (!res.ok) throw new Error("Correction failed");
        const result = await res.json();

        if (result.success) {
            closeEditModal();
            // Refresh current history page
            queryHistory(historyCurrentPage);
            // Refresh dashboard data to update stats and alerts list
            fetchDashboardData();
        } else {
            alert("修正失敗：" + result.error);
        }
    } catch (err) {
        alert("修正連線失敗，請檢查網路。");
        console.error("Submit correction error:", err);
    }
}

// ==========================================================================
// System Settings functions
// ==========================================================================

// Load camera settings
async function loadSystemSettings() {
    // Reset form to Add mode
    cancelCameraEdit();
    // Load list
    loadCamerasList();
    // Load Telegram settings
    loadTelegramSettings();
    // Load Web Auth settings
    loadAuthSettings();
    // Load System Maintenance settings
    loadMaintenanceSettings();
}

// Load and populate header select menu and settings list
async function loadCamerasList() {
    try {
        const res = await fetch('/api/cameras');
        if (!res.ok) throw new Error("Failed to load cameras list");
        const cameras = await res.json();
        
        // Update Video Stream Layout
        updateVideoStreamLayout(cameras);
        
        // 1. Update header dropdown selector
        const headerSelect = document.getElementById('header-camera-select');
        if (headerSelect) {
            let optionsHtml = '';
            const allSelectedAttr = currentCameraFilter === 'all' ? 'selected' : '';
            optionsHtml += `<option value="all" ${allSelectedAttr}>同時監測所有鏡頭</option>`;
            
            cameras.forEach(cam => {
                const selected = currentCameraFilter === String(cam.id) ? 'selected' : '';
                optionsHtml += `<option value="${cam.id}" ${selected}>${cam.name}</option>`;
            });
            headerSelect.innerHTML = optionsHtml;
        }
        
        // 2. Update settings tab table
        const tbody = document.getElementById('cameras-tbody');
        const emptyDiv = document.getElementById('cameras-empty');
        if (tbody) {
            if (cameras.length === 0) {
                tbody.innerHTML = '';
                emptyDiv.classList.remove('hidden');
                return;
            }
            emptyDiv.classList.add('hidden');
            
            let rowsHtml = '';
            cameras.forEach(cam => {
                const statusBadge = cam.is_active === 1 
                    ? `<span class="watch-badge vip" style="background-color: rgba(16,185,129,0.15); color: var(--color-primary); border: 1px solid rgba(16,185,129,0.3);">使用中</span>` 
                    : `<span class="watch-badge suspicious" style="background-color: rgba(255,255,255,0.05); color: var(--text-secondary); border: 1px solid rgba(255,255,255,0.1);">閒置</span>`;
                
                const activeBtn = cam.is_active === 1 
                    ? `<button class="btn btn-secondary btn-sm" onclick="toggleCameraActive(${cam.id}, 0)" style="padding: 4px 8px; font-size:11px; color: var(--color-yellow); border: 1px solid rgba(245,158,11,0.3); background: rgba(245,158,11,0.05);"><i class="fa-solid fa-pause"></i> 停用</button>` 
                    : `<button class="btn btn-secondary btn-sm" onclick="toggleCameraActive(${cam.id}, 1)" style="padding: 4px 8px; font-size:11px; color: var(--color-primary); border: 1px solid rgba(16,185,129,0.3); background: rgba(16,185,129,0.05);"><i class="fa-solid fa-play"></i> 啟用</button>`;
                
                // Truncate RTSP URL for cleaner look
                const displayUrl = cam.rtsp_url.length > 50 ? cam.rtsp_url.substring(0, 47) + '...' : cam.rtsp_url;
                
                rowsHtml += `
                    <tr>
                        <td><strong style="color: var(--text-primary); font-family: var(--font-tc);">${cam.name}</strong></td>
                        <td><code style="font-family: monospace; font-size: 11px; color: var(--text-secondary);" title="${cam.rtsp_url}">${displayUrl}</code></td>
                        <td>${statusBadge}</td>
                        <td>
                            <div style="display: flex; gap: 6px; align-items: center; justify-content: center;">
                                ${activeBtn}
                                <button class="btn btn-secondary btn-sm" onclick="editCamera(${cam.id}, '${cam.name}', '${cam.rtsp_url}')" style="padding: 4px 8px; font-size:11px; color: var(--color-blue);"><i class="fa-solid fa-pen"></i> 編輯</button>
                                <button class="btn-delete" onclick="deleteCamera(${cam.id}, '${cam.name}')" style="padding: 4px 8px; font-size:11px;"><i class="fa-solid fa-trash-can"></i> 刪除</button>
                            </div>
                        </td>
                    </tr>
                `;
            });
            tbody.innerHTML = rowsHtml;
        }
    } catch (err) {
        console.error("Load cameras list error:", err);
    }
}

// Save camera settings (Add/Edit)
async function saveCameraSettings(event) {
    event.preventDefault();
    const idInput = document.getElementById('settings-camera-id');
    const nameInput = document.getElementById('settings-camera-name');
    const urlInput = document.getElementById('settings-rtsp-url');

    const idVal = idInput.value;
    const nameVal = nameInput.value.trim();
    const urlVal = urlInput.value.trim();

    if (!nameVal || !urlVal) {
        alert("名稱或網址不能為空");
        return;
    }

    try {
        const payload = { name: nameVal, rtsp_url: urlVal };
        if (idVal) payload.id = parseInt(idVal);

        const res = await fetch('/api/cameras', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Save failed");
        const result = await res.json();

        if (result.success) {
            alert(idVal ? "已成功儲存並修改攝影機設定！" : "已成功新增攝影機！");
            cancelCameraEdit();
            loadCamerasList();
            fetchSystemStatus();
            
            // Force reload stream
            const streamImg = document.getElementById('stream-img');
            if (streamImg) {
                const currentSrc = streamImg.src.split('?')[0];
                streamImg.src = `${currentSrc}?t=${Date.now()}`;
            }
        } else {
            alert("儲存設定失敗：" + result.error);
        }
    } catch (err) {
        alert("儲存連線失敗，請檢查網路。");
        console.error("Save camera settings error:", err);
    }
}

function editCamera(id, name, rtspUrl) {
    document.getElementById('settings-camera-id').value = id;
    document.getElementById('settings-camera-name').value = name;
    document.getElementById('settings-rtsp-url').value = rtspUrl;
    
    document.getElementById('settings-form-title').innerHTML = `<i class="fa-solid fa-pen-to-square"></i> 修改攝影機設定`;
    document.getElementById('btn-cancel-edit-cam').classList.remove('hidden');
}

function cancelCameraEdit() {
    document.getElementById('settings-camera-id').value = '';
    document.getElementById('settings-camera-name').value = '';
    document.getElementById('settings-rtsp-url').value = '';
    
    document.getElementById('settings-form-title').innerHTML = `<i class="fa-solid fa-circle-plus"></i> 新增/修改攝影機`;
    document.getElementById('btn-cancel-edit-cam').classList.add('hidden');
}

async function deleteCamera(id, name) {
    if (!confirm(`確定要刪除攝影機「${name}」嗎？`)) {
        return;
    }

    try {
        const res = await fetch(`/api/cameras?id=${id}`, {
            method: 'DELETE'
        });

        if (!res.ok) throw new Error("DELETE failed");
        const result = await res.json();

        if (result.success) {
            loadCamerasList();
            fetchSystemStatus();
            
            const streamImg = document.getElementById('stream-img');
            if (streamImg) {
                const currentSrc = streamImg.src.split('?')[0];
                streamImg.src = `${currentSrc}?t=${Date.now()}`;
            }
        } else {
            alert("刪除失敗：" + result.error);
        }
    } catch (err) {
        alert("刪除連線失敗，請檢查網路。");
        console.error("Delete camera error:", err);
    }
}

async function changeActiveCamera(id) {
    try {
        const res = await fetch('/api/cameras/select', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ id: (id === 'all' || id === 'none') ? id : parseInt(id) })
        });

        if (!res.ok) throw new Error("Select failed");
        const result = await res.json();

        if (result.success) {
            lastFeedLatestId = 0;
            lastStatsJson = "";
            lastAlertsJson = "";
            loadCamerasList();
            fetchSystemStatus();
            fetchDashboardData();
        } else {
            alert("變更攝影機狀態失敗：" + result.error);
        }
    } catch (err) {
        alert("變更連線失敗，請確認網路。");
        console.error("Select camera error:", err);
    }
}

async function toggleCameraActive(id, isActive) {
    try {
        const res = await fetch('/api/cameras/select', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ id: parseInt(id), is_active: isActive })
        });

        if (!res.ok) throw new Error("Toggle failed");
        const result = await res.json();

        if (result.success) {
            lastFeedLatestId = 0;
            lastStatsJson = "";
            lastAlertsJson = "";
            loadCamerasList();
            fetchSystemStatus();
            fetchDashboardData();
        } else {
            alert("變更攝影機狀態失敗：" + result.error);
        }
    } catch (err) {
        alert("變更連線失敗，請確認網路。");
        console.error("Toggle camera error:", err);
    }
}

async function filterDashboardByCamera(value) {
    currentCameraFilter = value;
    lastFeedLatestId = 0;
    lastStatsJson = "";
    lastAlertsJson = "";
    await loadCamerasList();
    if (currentTab === 'history') {
        await queryHistory(0);
    } else {
        await fetchDashboardData();
    }
}

// ==========================================================================
// Telegram Notification Settings functions
// ==========================================================================

async function loadTelegramSettings() {
    try {
        const res = await fetch('/api/settings/telegram');
        if (!res.ok) throw new Error("Failed to load Telegram settings");
        const settings = await res.json();
        
        document.getElementById('tg-enabled').checked = settings.tg_enabled === '1';
        document.getElementById('tg-bot-token').value = settings.tg_bot_token || '';
        document.getElementById('tg-chat-id').value = settings.tg_chat_id || '';
    } catch (err) {
        console.error("Load Telegram settings error:", err);
    }
}

async function saveTelegramSettings(event) {
    event.preventDefault();
    
    const enabled = document.getElementById('tg-enabled').checked ? '1' : '0';
    const token = document.getElementById('tg-bot-token').value.trim();
    const chat_id = document.getElementById('tg-chat-id').value.trim();
    
    try {
        const res = await fetch('/api/settings/telegram', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tg_enabled: enabled,
                tg_bot_token: token,
                tg_chat_id: chat_id
            })
        });
        
        if (!res.ok) throw new Error("Save settings failed");
        const result = await res.json();
        
        if (result.success) {
            alert("Telegram 推播設定已成功儲存！");
            loadTelegramSettings();
        } else {
            alert("儲存設定失敗：" + result.error);
        }
    } catch (err) {
        alert("儲存連線失敗，請檢查網路。");
        console.error("Save Telegram settings error:", err);
    }
}

async function testTelegramNotification() {
    const token = document.getElementById('tg-bot-token').value.trim();
    const chat_id = document.getElementById('tg-chat-id').value.trim();
    
    if (!token || !chat_id) {
        alert("請先輸入 Telegram Bot Token 與 Chat ID 才能進行測試發送！");
        return;
    }
    
    const testBtn = document.querySelector("#telegram-settings-form button[type='button']");
    const originalHtml = testBtn.innerHTML;
    testBtn.disabled = true;
    testBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> 發送中...`;
    
    try {
        const res = await fetch('/api/settings/telegram/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                tg_bot_token: token,
                tg_chat_id: chat_id
            })
        });
        
        testBtn.disabled = false;
        testBtn.innerHTML = originalHtml;
        
        if (!res.ok) {
            const errResult = await res.json();
            throw new Error(errResult.error || "Test connection failed");
        }
        const result = await res.json();
        
        if (result.success) {
            alert("🔔 測試訊息已成功發送至您的 Telegram！請檢查您的手機。");
        } else {
            alert("測試發送失敗：" + result.error);
        }
    } catch (err) {
        testBtn.disabled = false;
        testBtn.innerHTML = originalHtml;
        alert("測試發送失敗！請確認 Token 和 Chat ID 是否正確，且已先在 Telegram 與該 Bot 對話。錯誤訊息: " + err.message);
        console.error("Test Telegram notification error:", err);
    }
}

// ==========================================================================
// Web Credentials Settings functions
// ==========================================================================

async function loadAuthSettings() {
    try {
        const res = await fetch('/api/settings/auth');
        if (!res.ok) throw new Error("Failed to load web credentials settings");
        const data = await res.json();
        
        document.getElementById('auth-username').value = data.web_username || 'admin';
        document.getElementById('auth-password').value = '';
    } catch (err) {
        console.error("Load Web Auth settings error:", err);
    }
}

async function saveAuthSettings(event) {
    event.preventDefault();
    
    const username = document.getElementById('auth-username').value.trim();
    const password = document.getElementById('auth-password').value;
    
    try {
        const res = await fetch('/api/settings/auth/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                web_username: username,
                web_password: password
            })
        });
        
        if (!res.ok) throw new Error("Save settings failed");
        const result = await res.json();
        
        if (result.success) {
            alert("網頁登入帳號密碼修改成功！新密碼已儲存生效。");
            loadAuthSettings();
        } else {
            alert("儲存密碼設定失敗：" + result.error);
        }
    } catch (err) {
        alert("儲存連線失敗，請檢查網路。");
        console.error("Save Web Auth settings error:", err);
    }
}

// ==========================================================================
// System Maintenance Settings functions
// ==========================================================================

async function loadMaintenanceSettings() {
    try {
        const res = await fetch('/api/settings/maintenance');
        if (!res.ok) throw new Error("Failed to load maintenance settings");
        const data = await res.json();
        
        document.getElementById('maintenance-retention-days').value = data.retention_days || 30;
        document.getElementById('maintenance-offline-threshold').value = data.offline_threshold_minutes || 5;
    } catch (err) {
        console.error("Load Maintenance settings error:", err);
    }
}

async function saveMaintenanceSettings(event) {
    event.preventDefault();
    
    const days = parseInt(document.getElementById('maintenance-retention-days').value);
    const threshold = parseInt(document.getElementById('maintenance-offline-threshold').value);
    
    if (isNaN(days) || days < 1) {
        alert("歷史資料保存天數必須是大於等於 1 的數字！");
        return;
    }
    if (isNaN(threshold) || threshold < 1) {
        alert("相機斷線告警時間門檻必須大於等於 1 的數字！");
        return;
    }
    
    try {
        const res = await fetch('/api/settings/maintenance/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                retention_days: days,
                offline_threshold_minutes: threshold
            })
        });
        
        if (!res.ok) throw new Error("Save maintenance settings failed");
        const result = await res.json();
        
        if (result.success) {
            alert("系統維運與清理設定已成功儲存！");
            loadMaintenanceSettings();
        } else {
            alert("儲存設定失敗：" + result.error);
        }
    } catch (err) {
        alert("儲存連線失敗，請檢查網路。");
        console.error("Save Maintenance settings error:", err);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Zone Editor – Interactive polygon gate zone editor
// Coordinate system: API/storage = 1920×1080 full-frame pixels
//                    SVG viewBox  = 640×360  (same as MJPEG stream size)
//                    scale factor = 3.0  (1920/640 = 1080/360 = 3)
// ═══════════════════════════════════════════════════════════════════════════
const ZONE_SCALE = 3.0;          // full-frame px per SVG px
const ZONE_VERTEX_R = 9;         // normal vertex circle radius (SVG px)
let zoneCurrentCamId = null;     // currently selected camera id (int)
let zonePolygonPts   = [];       // array of {x,y} in SVG (640×360) coords
let zoneDragIdx      = -1;       // index of vertex being dragged, or -1
let zoneDragOffX     = 0;
let zoneDragOffY     = 0;
let zoneActiveCameras = [];      // [{id, name}, ...]
let zoneSnapshotTimer = null;

// ── Init: populate camera dropdown then load first camera ──────────────────
async function zoneEditorInit() {
    try {
        const res = await fetch('/api/cameras');
        const cameras = await res.json();
        zoneActiveCameras = cameras;
        const sel = document.getElementById('zone-cam-select');
        sel.innerHTML = '';
        sel.style.cssText = 'color:#e2e8f0;background:#1a2234;border:1px solid rgba(0,220,255,0.4);padding:4px 10px;border-radius:6px;min-width:180px;';
        cameras.forEach(cam => {
            const opt = document.createElement('option');
            opt.value = cam.id;
            opt.textContent = `${cam.id} – ${cam.name}`;
            sel.appendChild(opt);
        });
        if (cameras.length > 0) {
            zoneEditorLoad(cameras[0].id);
        }
    } catch (e) {
        console.error('Zone editor init error:', e);
    }
}

// ── Load snapshot + zone polygon for a given camera id ────────────────────
async function zoneEditorLoad(camId) {
    zoneCurrentCamId = parseInt(camId);
    document.getElementById('zone-save-status').style.display = 'none';
    await zoneEditorRefreshSnapshot();
    await zoneEditorFetchPolygon(zoneCurrentCamId);
}

// ── Refresh camera snapshot image ─────────────────────────────────────────
async function zoneEditorRefreshSnapshot() {
    const img = document.getElementById('zone-snapshot');
    const cam = zoneCurrentCamId || 1;
    // Append timestamp to bust cache
    img.src = `/api/snapshot?cam_id=${cam}&t=${Date.now()}`;
}

// ── Fetch zone polygon from API and render ─────────────────────────────────
async function zoneEditorFetchPolygon(camId) {
    try {
        const res = await fetch('/api/zones');
        const data = await res.json();
        const key  = String(camId);
        if (data[key]) {
            // data[key] is array of [x1080, y1080]; convert to SVG coords
            zonePolygonPts = data[key].map(([fx, fy]) => ({
                x: fx / ZONE_SCALE,
                y: fy / ZONE_SCALE
            }));
        } else {
            // Default rectangle covering most of frame
            zonePolygonPts = [
                { x: 80,  y: 20  },
                { x: 560, y: 20  },
                { x: 560, y: 340 },
                { x: 80,  y: 340 }
            ];
        }
        zoneEditorRender();
    } catch (e) {
        console.error('Fetch zone error:', e);
    }
}

// ── Render polygon + draggable vertices into the SVG ──────────────────────
function zoneEditorRender() {
    const polygon = document.getElementById('zone-polygon');
    const vGroup  = document.getElementById('zone-vertices');

    // Update polygon points attribute
    polygon.setAttribute('points',
        zonePolygonPts.map(p => `${p.x},${p.y}`).join(' ')
    );

    // Rebuild vertex circles
    vGroup.innerHTML = '';
    zonePolygonPts.forEach((pt, idx) => {
        // Outer circle (hit-target + visual)
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', pt.x);
        circle.setAttribute('cy', pt.y);
        circle.setAttribute('r',  ZONE_VERTEX_R);
        circle.setAttribute('class', 'zone-vertex');
        circle.setAttribute('data-idx', idx);
        // Drag start
        circle.addEventListener('mousedown', zoneVertexMouseDown);
        circle.addEventListener('touchstart', zoneVertexTouchStart, { passive: false });
        vGroup.appendChild(circle);

        // Label (vertex number)
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', pt.x);
        label.setAttribute('y', pt.y);
        label.setAttribute('class', 'zone-vertex-label');
        label.textContent = idx + 1;
        vGroup.appendChild(label);
    });

    zoneEditorUpdateTable();
}

// ── Update coordinate table below the editor ──────────────────────────────
function zoneEditorUpdateTable() {
    const tbody = document.getElementById('zone-coord-tbody');
    tbody.innerHTML = '';
    zonePolygonPts.forEach((pt, idx) => {
        const fx = Math.round(pt.x * ZONE_SCALE);
        const fy = Math.round(pt.y * ZONE_SCALE);
        tbody.innerHTML += `
            <tr>
                <td>頂點 ${idx + 1}</td>
                <td>${fx}</td>
                <td>${fy}</td>
            </tr>`;
    });
}

// ── Mouse drag handlers ────────────────────────────────────────────────────
function zoneVertexMouseDown(e) {
    e.preventDefault();
    const idx = parseInt(e.target.getAttribute('data-idx'));
    const svgRect = document.getElementById('zone-svg').getBoundingClientRect();
    const svgW = svgRect.width;
    const svgH = svgRect.height;
    // Scale from screen px to SVG viewBox coords
    const scaleX = 640 / svgW;
    const scaleY = 360 / svgH;

    zoneDragIdx = idx;
    e.target.classList.add('dragging');

    function onMove(ev) {
        const x = Math.max(0, Math.min(640, (ev.clientX - svgRect.left) * scaleX));
        const y = Math.max(0, Math.min(360, (ev.clientY - svgRect.top)  * scaleY));
        zonePolygonPts[zoneDragIdx] = { x, y };
        zoneEditorRender();
    }
    function onUp() {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        document.querySelectorAll('.zone-vertex').forEach(c => c.classList.remove('dragging'));
        zoneDragIdx = -1;
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
}

// ── Touch drag handlers ────────────────────────────────────────────────────
function zoneVertexTouchStart(e) {
    e.preventDefault();
    const idx = parseInt(e.target.getAttribute('data-idx'));
    const svgRect = document.getElementById('zone-svg').getBoundingClientRect();
    const scaleX = 640 / svgRect.width;
    const scaleY = 360 / svgRect.height;
    zoneDragIdx = idx;

    function onMove(ev) {
        const touch = ev.touches[0];
        const x = Math.max(0, Math.min(640, (touch.clientX - svgRect.left) * scaleX));
        const y = Math.max(0, Math.min(360, (touch.clientY - svgRect.top)  * scaleY));
        zonePolygonPts[zoneDragIdx] = { x, y };
        zoneEditorRender();
    }
    function onEnd() {
        document.removeEventListener('touchmove', onMove);
        document.removeEventListener('touchend',  onEnd);
        zoneDragIdx = -1;
    }
    document.addEventListener('touchmove', onMove, { passive: false });
    document.addEventListener('touchend',  onEnd);
}

// ── Save current polygon to backend (hot-reload, no restart needed) ────────
async function zoneEditorSave() {
    if (zoneCurrentCamId === null) return;
    // Convert SVG coords back to 1920×1080 full-frame coords
    const polygon = zonePolygonPts.map(p => [
        Math.round(p.x * ZONE_SCALE),
        Math.round(p.y * ZONE_SCALE)
    ]);
    try {
        const res = await fetch('/api/zones', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ cam_id: zoneCurrentCamId, polygon })
        });
        const data = await res.json();
        if (data.success) {
            const status = document.getElementById('zone-save-status');
            status.style.display = 'inline';
            setTimeout(() => { status.style.display = 'none'; }, 3000);
        } else {
            alert('儲存失敗：' + data.error);
        }
    } catch (e) {
        alert('連線失敗，請確認引擎運作中。');
        console.error('Zone save error:', e);
    }
}

// ── Reset current camera zone to server-stored value ──────────────────────
async function zoneEditorReset() {
    if (zoneCurrentCamId === null) return;
    await zoneEditorFetchPolygon(zoneCurrentCamId);
    document.getElementById('zone-save-status').style.display = 'none';
}
