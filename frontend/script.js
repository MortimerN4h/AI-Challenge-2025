/*
File: script.js
Mô tả: Xử lý logic phía client cho hệ thống truy vấn.
*/

// --- Lấy các phần tử HTML ---
const searchButton = document.getElementById('searchButton');
const searchQueryInput = document.getElementById('searchQuery');
const resultsContainer = document.getElementById('results-container');
const loadingIndicator = document.getElementById('loading');
const resultsInfo = document.getElementById('results-info');

const API_SEARCH_URL = `/api/search`;
const KEYFRAMES_BASE_URL = `/keyframes/`;

/**
 * Hàm chính để thực hiện tìm kiếm.
 */
async function performSearch() {
    const query = searchQueryInput.value.trim();
    if (!query) return;

    setLoading(true);
    resultsContainer.innerHTML = '';
    resultsInfo.innerHTML = '';

    try {
        const startTime = Date.now();
        const response = await fetch(API_SEARCH_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, k: 48 })
        });
        const endTime = Date.now();

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data.results, (endTime - startTime) / 1000);

    } catch (error) {
        console.error("Lỗi khi tìm kiếm:", error);
        resultsInfo.innerHTML = `<span style="color: red;">Lỗi: ${error.message}</span>`;
    } finally {
        setLoading(false);
    }
}

/**
 * Hiển thị kết quả.
 */
function displayResults(results, duration) {
    if (results.length === 0) {
        resultsInfo.innerHTML = 'Không tìm thấy kết quả phù hợp.';
        return;
    }

    resultsInfo.innerHTML = `Tìm thấy ${results.length} kết quả trong ${duration.toFixed(2)} giây.`;

    results.forEach(item => {
        const itemDiv = document.createElement('div');
        itemDiv.className = 'result-item';

        const img = document.createElement('img');
        img.src = KEYFRAMES_BASE_URL + item.frame_id;
        img.alt = item.frame_id;
        img.loading = 'lazy';
        img.onerror = function() {
            this.src = 'data:image/svg+xml;charset=UTF-8,<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 250 160"><rect width="100%" height="100%" fill="%23eee"/><text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif" font-size="14" fill="%23777">Image Not Found</text></svg>';
        };

        const info = document.createElement('div');
        info.className = 'result-info';
        info.innerHTML = `ID: ${item.frame_id}<br>Score: ${item.score.toFixed(4)}`;

        itemDiv.appendChild(img);
        itemDiv.appendChild(info);
        resultsContainer.appendChild(itemDiv);
    });
}

/**
 * Quản lý trạng thái loading.
 */
function setLoading(isLoading) {
    loadingIndicator.style.display = isLoading ? 'block' : 'none';
    searchButton.disabled = isLoading;
    searchQueryInput.disabled = isLoading;
}

// --- Gán sự kiện ---
searchButton.addEventListener('click', performSearch);
searchQueryInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        performSearch();
    }
});