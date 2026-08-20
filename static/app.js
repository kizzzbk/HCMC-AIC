// Global State
let currentQuery = "";
let currentOffset = 0;
let currentTopK = 20;
let totalResults = 0;
let pendingSubmitGlobalId = null;
let currentResultsList = [];
let currentActiveView = "search";
let allVideosData = [];
let selectedDatasetFilter = "";

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    checkSystemHealth();
    loadVideosList();

    // Keybindings
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") {
            closeNeighborModal();
            closeSubmitModal();
        }
    });
});

// 1. View Switcher (Search vs Video Gallery)
function switchView(viewName) {
    currentActiveView = viewName;
    const tabSearch = document.getElementById("tab-search");
    const tabVideos = document.getElementById("tab-videos");
    const searchViewContainer = document.getElementById("search-view-container");
    const videoGallerySection = document.getElementById("video-gallery-section");

    if (viewName === "videos") {
        tabSearch.classList.remove("active");
        tabVideos.classList.add("active");
        searchViewContainer.style.display = "none";
        videoGallerySection.style.display = "block";

        if (allVideosData.length === 0) {
            loadDetailedVideos();
        }
    } else {
        tabVideos.classList.remove("active");
        tabSearch.classList.add("active");
        videoGallerySection.style.display = "none";
        searchViewContainer.style.display = "block";
    }
}

// 2. Health Check
async function checkSystemHealth() {
    try {
        const res = await fetch("/api/health");
        const data = await res.json();
        const statusElem = document.getElementById("system-status");
        const textElem = document.getElementById("status-text");
        if (data.status === "online") {
            statusElem.style.borderColor = "rgba(16, 185, 129, 0.4)";
            textElem.innerText = `Sẵn sàng (${data.total_frames.toLocaleString()} frames)`;
        }
    } catch (err) {
        const textElem = document.getElementById("status-text");
        textElem.innerText = "Mất kết nối server";
        textElem.parentElement.style.borderColor = "rgba(244, 63, 94, 0.4)";
    }
}

// 3. Load Video List for Filters
async function loadVideosList() {
    try {
        const res = await fetch("/api/videos");
        const data = await res.json();
        const select = document.getElementById("video-filter");
        if (data.videos && data.videos.length > 0) {
            select.innerHTML = '<option value="">Tất cả video</option>';
            data.videos.forEach(v => {
                const opt = document.createElement("option");
                opt.value = v;
                opt.textContent = v;
                select.appendChild(opt);
            });

            // Update badge count
            const badge = document.getElementById("video-total-badge");
            if (badge) {
                badge.innerText = `${data.videos.length} videos`;
            }
            const countAll = document.getElementById("count-all");
            if (countAll) {
                countAll.innerText = data.videos.length;
            }

            // Tạo dataset pills động từ prefix của video_id (L21, L22, L25...)
            const datasetCounts = {};
            data.videos.forEach(v => {
                const prefix = v.match(/^(L\d+)/)?.[1];
                if (prefix) datasetCounts[prefix] = (datasetCounts[prefix] || 0) + 1;
            });

            const pillsContainer = document.getElementById("dataset-pills");
            if (pillsContainer) {
                // Giữ nút "Tất cả", xóa pills cũ
                pillsContainer.innerHTML = `
                    <button class="pill-btn active" data-dataset="" onclick="filterVideoGallery('')">
                        Tất cả (<span id="count-all">${data.videos.length}</span>)
                    </button>
                `;
                Object.entries(datasetCounts).sort().forEach(([ds, count]) => {
                    const btn = document.createElement("button");
                    btn.className = "pill-btn";
                    btn.setAttribute("data-dataset", ds);
                    btn.setAttribute("onclick", `filterVideoGallery('${ds}')`);
                    btn.textContent = `${ds} (${count})`;
                    pillsContainer.appendChild(btn);
                });
            }
        }
    } catch (err) {
        console.error("Failed to load videos:", err);
    }
}

// 4. Load Detailed Videos with Thumbnails
async function loadDetailedVideos() {
    const grid = document.getElementById("video-cards-grid");
    grid.innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">⏳</div>
            <h3>Đang tải danh sách video & thumbnail...</h3>
            <p>Hệ thống đang chuẩn bị keyframe preview đại diện cho các video.</p>
        </div>
    `;

    try {
        const res = await fetch("/api/videos?detailed=true");
        if (!res.ok) throw new Error("Không thể tải danh sách video chi tiết");
        const data = await res.json();
        allVideosData = data.videos || [];
        renderVideoGallery(allVideosData);
    } catch (err) {
        grid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">❌</div>
                <h3>Lỗi khi tải video gallery</h3>
                <p>${err.message}</p>
            </div>
        `;
    }
}

// 5. Filter Video Gallery by Dataset
function filterVideoGallery(dataset) {
    selectedDatasetFilter = dataset;

    // Update pill buttons active state
    const pills = document.querySelectorAll(".dataset-pills .pill-btn");
    pills.forEach(p => {
        if (p.getAttribute("data-dataset") === dataset) {
            p.classList.add("active");
        } else {
            p.classList.remove("active");
        }
    });

    if (!dataset) {
        renderVideoGallery(allVideosData);
    } else {
        const filtered = allVideosData.filter(v => v.dataset === dataset || v.video_id.startsWith(dataset));
        renderVideoGallery(filtered);
    }
}

// 6. Render Video Gallery Cards
function renderVideoGallery(videos) {
    const grid = document.getElementById("video-cards-grid");
    grid.innerHTML = "";

    if (!videos || videos.length === 0) {
        grid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">🎬</div>
                <h3>Không tìm thấy video nào</h3>
                <p>Không có video phù hợp với bộ lọc hiện tại.</p>
            </div>
        `;
        return;
    }

    videos.forEach(v => {
        const card = document.createElement("div");
        card.className = "video-card glass-card";

        const dsClass = `badge-ds-${v.dataset.toLowerCase()}`;
        const thumbnailSrc = v.image_url || (v.thumbnail_path ? `/media/${v.thumbnail_path}` : "/media/placeholder.svg");

        // Build sample micro-thumbnails HTML
        let samplesHtml = "";
        if (v.sample_frames && v.sample_frames.length > 1) {
            samplesHtml = `
                <div class="video-samples-row">
                    ${v.sample_frames.slice(1, 4).map(s => `
                        <div class="micro-thumbnail-wrap" title="Frame #${s.global_id} (idx:${s.frame_idx})">
                            <img src="${s.image_url}" alt="${s.frame_id}" loading="lazy" onerror="this.src='/media/placeholder.svg'">
                        </div>
                    `).join("")}
                </div>
            `;
        }

        card.innerHTML = `
            <div class="video-thumbnail-wrap">
                <img src="${thumbnailSrc}" alt="${v.video_id}" loading="lazy" onerror="this.src='/media/placeholder.svg'">
                <span class="video-dataset-badge ${dsClass}">${v.dataset}</span>
                <span class="video-frames-badge">📹 ${v.total_frames.toLocaleString()} frames</span>
            </div>
            <div class="video-card-body">
                <div class="video-card-header">
                    <h3 class="video-title">${v.video_id}</h3>
                    <span class="video-meta-tag">${v.dataset} Dataset</span>
                </div>
                ${samplesHtml}
                <div class="video-card-actions">
                    <button class="btn btn-outline btn-sm" onclick="selectVideoToSearch('${v.video_id}')">
                        <span>🔍 Lọc Video Này</span>
                    </button>
                    ${v.sample_frames && v.sample_frames.length > 0 ? `
                        <button class="btn btn-secondary btn-sm" onclick="openNeighborModal(${v.sample_frames[0].global_id})">
                            <span>🎞 Preview ±5</span>
                        </button>
                    ` : ''}
                </div>
            </div>
        `;

        grid.appendChild(card);
    });
}

// 7. Select video to search / explore frames
function selectVideoToSearch(videoId) {
    // Select in dropdown
    const select = document.getElementById("video-filter");
    select.value = videoId;

    // Switch view
    switchView("search");

    // If query input is empty, fill a hint or prompt user
    const queryInput = document.getElementById("query-input");
    if (!queryInput.value.trim()) {
        showToast(`Đã chọn bộ lọc video: ${videoId}. Nhập từ khóa để tìm kiếm trong video này.`, "info");
        queryInput.focus();
    } else {
        handleSearch();
    }
}

// 8. Handle Search
async function handleSearch(e) {
    if (e) e.preventDefault();
    const queryInput = document.getElementById("query-input");
    const query = queryInput.value.trim();
    if (!query) return;

    currentQuery = query;
    currentOffset = 0;
    currentTopK = parseInt(document.getElementById("top-k-select").value) || 20;

    const selectedVideo = document.getElementById("video-filter").value;
    const video_ids = selectedVideo ? [selectedVideo] : null;

    const searchBtn = document.getElementById("search-btn");
    searchBtn.disabled = true;
    searchBtn.innerHTML = "<span>Đang tìm...</span>";

    try {
        const payload = {
            query: currentQuery,
            top_k: currentTopK,
            offset: 0,
            video_ids: video_ids
        };

        const res = await fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errData = await res.json();
            throw new Error(errData.detail || "Search failed");
        }

        const data = await res.json();
        totalResults = data.total;
        currentResultsList = data.results;

        // Update stats
        document.getElementById("stats-bar").style.display = "flex";
        document.getElementById("total-count").innerText = totalResults.toLocaleString();
        document.getElementById("current-query-display").innerText = currentQuery;
        document.getElementById("query-time").innerText = data.time_taken_ms;
        document.getElementById("page-indicator").innerText = `Hiển thị 1 - ${Math.min(currentTopK, totalResults)}`;

        // Render Results
        renderResults(data.results, false);

        // Update pagination visibility
        const paginationPanel = document.getElementById("pagination-panel");
        if (totalResults > currentTopK) {
            paginationPanel.style.display = "flex";
        } else {
            paginationPanel.style.display = "none";
        }

    } catch (err) {
        showToast(err.message, "error");
    } finally {
        searchBtn.disabled = false;
        searchBtn.innerHTML = "<span>Tìm kiếm</span>";
    }
}

// 9. Next Top-K Pagination
async function loadNextTopK() {
    currentOffset += currentTopK;
    const loadMoreBtn = document.getElementById("load-more-btn");
    loadMoreBtn.disabled = true;
    loadMoreBtn.innerHTML = "<span>Đang tải...</span>";

    const selectedVideo = document.getElementById("video-filter").value;
    const video_ids = selectedVideo ? [selectedVideo] : null;

    try {
        const payload = {
            query: currentQuery,
            top_k: currentTopK,
            offset: currentOffset,
            video_ids: video_ids
        };

        const res = await fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.results && data.results.length > 0) {
            currentResultsList = currentResultsList.concat(data.results);
            renderResults(data.results, true);

            const showingCount = Math.min(currentOffset + currentTopK, totalResults);
            document.getElementById("page-indicator").innerText = `Hiển thị 1 - ${showingCount}`;

            if (showingCount >= totalResults) {
                document.getElementById("pagination-panel").style.display = "none";
            }
        } else {
            document.getElementById("pagination-panel").style.display = "none";
            showToast("Đã hiển thị toàn bộ kết quả phù hợp.", "info");
        }
    } catch (err) {
        showToast("Lỗi khi tải thêm kết quả: " + err.message, "error");
    } finally {
        loadMoreBtn.disabled = false;
        loadMoreBtn.innerHTML = "<span>Tải thêm kết quả (Next Top-K) ⬇</span>";
    }
}

// 10. Render Keyframe Search Result Grid
function renderResults(items, append = false) {
    const grid = document.getElementById("results-grid");
    if (!append) grid.innerHTML = "";

    if (items.length === 0 && !append) {
        grid.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">❌</div>
                <h3>Không tìm thấy kết quả phù hợp</h3>
                <p>Thử thay đổi từ khóa hoặc xóa bộ lọc để mở rộng phạm vi tìm kiếm.</p>
            </div>
        `;
        return;
    }

    items.forEach(item => {
        const card = document.createElement("div");
        card.className = "keyframe-card";

        const imageUrl = item.image_url || `/media/${item.image_path}`;

        card.innerHTML = `
            <div class="card-image-wrap">
                <img src="${imageUrl}" alt="Global ID ${item.global_id}" loading="lazy" onerror="this.src='/media/placeholder.svg'">
                <span class="card-id-badge">ID: ${item.global_id}</span>
                <span class="card-score-badge">Score: ${item.score}</span>
            </div>
            <div class="card-body">
                <div class="card-title">Global ID: #${item.global_id}</div>
                <div class="card-meta-row">
                    <span>📹 ${item.video_id}</span>
                    <span>⏱ ${item.timestamp}s (f:${item.frame_idx})</span>
                </div>
                <div class="card-meta-row" style="font-size:0.75rem; color:var(--text-muted); margin-top:2px;">
                    <span>Frame: ${item.frame_id}</span>
                </div>
                <div class="card-actions">
                    <button class="card-btn" onclick="openNeighborModal(${item.global_id})">
                        🔍 Neighbors ±5
                    </button>
                    <button class="card-btn btn-card-submit" onclick="openSubmitModal(${item.global_id})">
                        🚀 Submit (${item.global_id})
                    </button>
                </div>
            </div>
        `;
        grid.appendChild(card);
    });
}

// 11. Neighbor Frames Modal
async function openNeighborModal(global_id) {
    const modal = document.getElementById("neighbor-modal");
    const timeline = document.getElementById("timeline-list");
    timeline.innerHTML = "<p style='color: var(--text-secondary); padding: 2rem;'>Đang tải danh sách khung hình...</p>";
    modal.classList.add("active");

    try {
        const res = await fetch(`/api/frames/${global_id}/neighbors?window=5`);
        if (!res.ok) throw new Error("Không thể lấy neighbor frames");
        const data = await res.json();

        document.getElementById("neighbor-subtitle").innerText = 
            `Video: ${data.video_id} | Center Global ID: #${data.center_global_id} (frame_idx: ${data.center_frame_idx}) | Total: ${data.total_neighbors} frames`;

        timeline.innerHTML = "";
        data.neighbors.forEach(f => {
            const itemDiv = document.createElement("div");
            itemDiv.className = `timeline-item ${f.is_center ? 'center-frame' : ''}`;
            const imgUrl = f.image_url || `/media/${f.image_path}`;

            itemDiv.innerHTML = `
                <div class="timeline-img-wrap">
                    <img src="${imgUrl}" alt="Global ID ${f.global_id}" loading="lazy" onerror="this.src='/media/placeholder.svg'">
                </div>
                <div class="timeline-info">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="timeline-badge">${f.is_center ? '⭐ CENTER' : 'Idx ' + f.frame_idx}</span>
                        <span style="color:var(--accent-sky); font-weight:bold; font-size:0.8rem;">#${f.global_id}</span>
                    </div>
                    <span style="color:var(--text-secondary); margin-top:2px;">⏱ ${f.timestamp}s</span>
                    <button class="card-btn btn-card-submit" style="margin-top:6px; padding:4px;" onclick="event.stopPropagation(); openSubmitModal(${f.global_id})">
                        Submit #${f.global_id}
                    </button>
                </div>
            `;
            timeline.appendChild(itemDiv);
        });

    } catch (err) {
        timeline.innerHTML = `<p style="color: var(--accent-rose); padding: 1rem;">Lỗi: ${err.message}</p>`;
    }
}

function closeNeighborModal() {
    document.getElementById("neighbor-modal").classList.remove("active");
}

// 12. Submit Modal & Handler
function openSubmitModal(global_id) {
    pendingSubmitGlobalId = global_id;
    const modal = document.getElementById("submit-modal");
    const body = document.getElementById("submit-modal-body");

    // Find item if in current results
    const item = currentResultsList.find(r => r.global_id === global_id);
    const videoId = item ? item.video_id : "Video";
    const frameId = item ? item.frame_id : `frame_${global_id}`;
    const frameIdx = item ? item.frame_idx : 0;
    const timestamp = item ? item.timestamp : 0;

    body.innerHTML = `
        <div style="background:rgba(15,23,42,0.8); padding:1.25rem; border-radius:8px; border:1px solid var(--border-color); display:flex; flex-direction:column; gap:0.6rem; font-size:0.95rem;">
            <div style="font-size:1.15rem; color:var(--accent-sky); border-bottom:1px solid var(--border-color); padding-bottom:0.5rem;">
                <strong>Global ID (Nộp BTC):</strong> <span style="font-weight:bold; color:#38bdf8;">#${global_id}</span>
            </div>
            <div><strong>Video ID:</strong> <span>${videoId}</span></div>
            <div><strong>Frame Index:</strong> <span>${frameIdx}</span></div>
            <div><strong>Frame ID:</strong> <span style="color:var(--text-muted); font-size:0.85rem;">${frameId}</span></div>
            <div><strong>Timestamp:</strong> <span>${timestamp}s</span></div>
            <div><strong>Query:</strong> <span style="color:var(--text-secondary); font-style:italic;">"${currentQuery || 'None'}"</span></div>
        </div>
        <p style="margin-top:1rem; font-size:0.85rem; color:var(--text-secondary);">
            Hệ thống sẽ đóng gói chính xác <strong>global_id: ${global_id}</strong> theo định dạng chuẩn BTC để gửi đến server đánh giá.
        </p>
    `;

    modal.classList.add("active");
}

function closeSubmitModal() {
    document.getElementById("submit-modal").classList.remove("active");
    pendingSubmitGlobalId = null;
}

async function executeSubmission() {
    if (!pendingSubmitGlobalId) return;

    const btn = document.getElementById("confirm-submit-btn");
    btn.disabled = true;
    btn.innerHTML = "<span>Đang nộp...</span>";

    try {
        const payload = {
            global_id: pendingSubmitGlobalId,
            query: currentQuery
        };

        const res = await fetch("/api/submit", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (data.success) {
            closeSubmitModal();
            showToast(`Nộp thành công frame #${pendingSubmitGlobalId}! (BTC Status: 200 OK)`, "success");
        } else {
            throw new Error(data.detail || "Submit thất bại");
        }
    } catch (err) {
        showToast(`Lỗi khi nộp bài: ${err.message}`, "error");
    } finally {
        btn.disabled = false;
        btn.innerHTML = "<span>Xác nhận Nộp</span>";
    }
}

// 13. Utility Functions
function clearSearch() {
    document.getElementById("query-input").value = "";
    document.getElementById("stats-bar").style.display = "none";
    document.getElementById("pagination-panel").style.display = "none";
    document.getElementById("results-grid").innerHTML = `
        <div class="empty-state">
            <div class="empty-icon">🎯</div>
            <h3>Sẵn sàng truy vấn Video Keyframe</h3>
            <p>Nhập mô tả hành động, đối tượng hoặc bối cảnh để tìm kiếm keyframe chính xác.</p>
        </div>
    `;
    currentQuery = "";
    currentOffset = 0;
}

function onTopKChange() {
    if (currentQuery) {
        handleSearch();
    }
}

function showToast(message, type = "info") {
    const toast = document.getElementById("toast");
    toast.innerText = message;
    toast.className = `toast show toast-${type}`;
    setTimeout(() => {
        toast.className = "toast";
    }, 3500);
}
