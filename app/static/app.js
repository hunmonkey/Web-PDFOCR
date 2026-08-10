// =========================================================================
// PDFOCR 화면 동작
//
// 흐름: 파일 고르기 → 변환 시작(POST) → 진행 상황 받기(SSE) → 결과 내려받기
// =========================================================================

(function () {
    "use strict";

    const el = {
        dropzone: document.getElementById("dropzone"),
        fileInput: document.getElementById("file-input"),
        filePick: document.getElementById("file-pick"),
        pickedFile: document.getElementById("picked-file"),
        stageView: document.getElementById("stage-view"),
        railFile: document.getElementById("rail-file"),
        railOcr: document.getElementById("rail-ocr"),
        railFix: document.getElementById("rail-fix"),
        railExport: document.getElementById("rail-export"),
        toolFile: document.getElementById("tool-file"),
        toolOcr: document.getElementById("tool-ocr"),
        toolFix: document.getElementById("tool-fix"),
        toolExport: document.getElementById("tool-export"),
        logoutButton: document.getElementById("logout-button"),
        tabUploadButton: document.getElementById("tab-upload-button"),
        tabStorageButton: document.getElementById("tab-storage-button"),
        tabUpload: document.getElementById("tab-upload"),
        tabStorage: document.getElementById("tab-storage"),
        storageList: document.getElementById("storage-list"),
        storageHint: document.getElementById("storage-hint"),
        storageRefresh: document.getElementById("storage-refresh"),
        processorServer: document.getElementById("processor-server"),
        processorBrowser: document.getElementById("processor-browser"),
        processorBrowserChoice: document.getElementById("processor-browser-choice"),
        browserProcessorHint: document.getElementById("browser-processor-hint"),
        dpi: document.getElementById("dpi"),
        dpiDescription: document.getElementById("dpi-description"),
        textScore: document.getElementById("text-score"),
        modeKeep: document.getElementById("mode-keep"),
        modeRebuild: document.getElementById("mode-rebuild"),
        modeRebuildChoice: document.getElementById("mode-rebuild-choice"),
        autoRotate: document.getElementById("auto-rotate"),
        autoRotateChoice: document.getElementById("auto-rotate-choice"),
        deskew: document.getElementById("deskew"),
        deskewChoice: document.getElementById("deskew-choice"),
        correctionNote: document.getElementById("correction-note"),
        exportFormats: document.querySelectorAll(".export-format"),
        textLayerWarning: document.getElementById("text-layer-warning"),
        exportButtons: document.getElementById("export-buttons"),
        exportError: document.getElementById("export-error"),
        readyHint: document.getElementById("ready-hint"),
        startButton: document.getElementById("start-button"),
        stopButton: document.getElementById("stop-button"),
        statusBadge: document.getElementById("status-badge"),
        errorBox: document.getElementById("error-box"),
        progressCard: document.getElementById("progress-card"),
        progressPercent: document.getElementById("progress-percent"),
        progressBadge: document.getElementById("progress-badge"),
        progressBar: document.getElementById("progress-bar"),
        progressMessage: document.getElementById("progress-message"),
        resultCard: document.getElementById("result-card"),
        resultHint: document.getElementById("result-hint"),
        resultGrid: document.getElementById("result-grid"),
        downloadButton: document.getElementById("download-button"),
        resetButton: document.getElementById("reset-button"),
    };

    // 상태 이름에 맞는 배지 모양. 색만으로 구분하지 않도록 글자도 함께 바꾼다.
    const BADGE_CLASS = {
        "대기": "badge-waiting",
        "실행 중": "badge-running",
        "성공": "badge-done",
        "실패": "badge-failed",
        "취소": "badge-canceled",
    };

    // 어떤 장치로 처리했는지 쉬운 말로 바꿔 보여 준다.
    const DEVICE_LABEL = {
        "CUDA": "그래픽카드",
        "DirectML": "그래픽카드",
        "CPU": "중앙처리장치",
        "WebGPU": "이 컴퓨터의 그래픽카드",
    };
    const DEVICE_HINT = {
        "CUDA": "NVIDIA 가속 사용",
        "DirectML": "DirectML 가속 사용",
        "CPU": "그래픽 가속을 쓰지 못해 CPU로 처리",
        "WebGPU": "브라우저 WebGPU 가속 사용",
    };

    let pickedFile = null;      // 내 컴퓨터에서 고른 파일
    let pickedStored = null;    // 보관함에서 고른 파일 {relative, name, size}
    let currentJobId = null;
    let currentProcessor = "server";
    let eventSource = null;
    let createController = null;
    let browserController = null;
    let browserSupported = false;
    let pendingJobRequest = null;

    // ── 도우미 ───────────────────────────────────────────────────────────
    function show(node) { node.classList.remove("is-hidden"); }
    function hide(node) { node.classList.add("is-hidden"); }

    function formatSize(bytes) {
        if (bytes >= 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
        return Math.round(bytes / 1024) + " KB";
    }

    function setBadge(node, status) {
        node.className = "badge " + (BADGE_CLASS[status] || "badge-waiting");
        node.textContent = status;
    }

    function showError(message) {
        el.errorBox.textContent = message;
        show(el.errorBox);
    }

    function clearError() {
        el.errorBox.textContent = "";
        hide(el.errorBox);
    }

    function closeProgressConnection() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }

    function abortBrowserWork() {
        if (browserController) {
            browserController.abort();
            browserController = null;
        }
    }

    function abortCreateRequest() {
        if (createController) {
            createController.abort();
            createController = null;
        }
    }

    function stopJobInBackground(jobId) {
        if (!jobId) return;
        const queued = navigator.sendBeacon("/api/jobs/" + jobId + "/stop");
        if (!queued) {
            void fetch("/api/jobs/" + jobId + "/stop", {
                method: "POST",
                keepalive: true,
            });
        }
    }

    function discardPendingJobRequest() {
        if (!pendingJobRequest) return;
        abortCreateRequest();
        stopJobInBackground(pendingJobRequest.id);
        if (currentJobId === pendingJobRequest.id) currentJobId = null;
        pendingJobRequest = null;
    }

    async function cancelPendingJobRequest() {
        if (!pendingJobRequest) return;
        const request = pendingJobRequest;
        abortCreateRequest();
        const response = await fetchWithTimeout(
            "/api/jobs/" + request.id + "/stop",
            { method: "POST" },
            null,
            5000
        );
        if (!response.ok && response.status !== 404) {
            throw new Error(await responseMessage(response, "이전 변환 요청을 정리하지 못했습니다."));
        }
        if (pendingJobRequest === request) pendingJobRequest = null;
        if (currentJobId === request.id) currentJobId = null;
    }

    function newRequestId() {
        const bytes = new Uint8Array(16);
        crypto.getRandomValues(bytes);
        return Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
    }

    function retryableStatus(status) {
        return status === 408 || status === 425 || status === 429 || status >= 500;
    }

    function wait(milliseconds, signal) {
        return new Promise((resolve, reject) => {
            if (signal && signal.aborted) {
                reject(new DOMException("사용자가 작업을 중지했습니다.", "AbortError"));
                return;
            }
            const finish = () => {
                if (signal) signal.removeEventListener("abort", onAbort);
                resolve();
            };
            const timer = window.setTimeout(finish, milliseconds);
            const onAbort = () => {
                window.clearTimeout(timer);
                reject(new DOMException("사용자가 작업을 중지했습니다.", "AbortError"));
            };
            if (signal) signal.addEventListener("abort", onAbort, { once: true });
        });
    }

    async function fetchWithTimeout(url, options, outerSignal, timeoutMilliseconds = 15000) {
        const controller = new AbortController();
        const onAbort = () => controller.abort();
        if (outerSignal) {
            if (outerSignal.aborted) {
                throw new DOMException("사용자가 작업을 중지했습니다.", "AbortError");
            }
            outerSignal.addEventListener("abort", onAbort, { once: true });
        }
        const timer = window.setTimeout(() => controller.abort(), timeoutMilliseconds);
        try {
            return await fetch(url, { ...options, signal: controller.signal });
        } catch (error) {
            if (outerSignal && outerSignal.aborted) {
                throw new DOMException("사용자가 작업을 중지했습니다.", "AbortError");
            }
            if (controller.signal.aborted) {
                const seconds = Math.round(timeoutMilliseconds / 1000);
                throw new Error("서버 응답을 " + seconds + "초 안에 받지 못했습니다.");
            }
            throw error;
        } finally {
            window.clearTimeout(timer);
            if (outerSignal) outerSignal.removeEventListener("abort", onAbort);
        }
    }

    async function fetchJsonWithTimeout(
        url,
        options,
        outerSignal,
        timeoutMilliseconds = 15000
    ) {
        const controller = new AbortController();
        const onAbort = () => controller.abort();
        if (outerSignal) {
            if (outerSignal.aborted) {
                throw new DOMException("사용자가 작업을 중지했습니다.", "AbortError");
            }
            outerSignal.addEventListener("abort", onAbort, { once: true });
        }
        const timer = window.setTimeout(() => controller.abort(), timeoutMilliseconds);
        try {
            const response = await fetch(url, { ...options, signal: controller.signal });
            let data = {};
            try {
                data = await response.json();
            } catch (error) {
                // 성공 응답의 본문이 끊기거나 깨졌다면 request_id로 다시 확인해야 한다.
                if (response.ok) throw error;
            }
            return { response, data };
        } catch (error) {
            if (outerSignal && outerSignal.aborted) {
                throw new DOMException("사용자가 작업을 중지했습니다.", "AbortError");
            }
            if (controller.signal.aborted) {
                const seconds = Math.round(timeoutMilliseconds / 1000);
                throw new Error("서버 응답을 " + seconds + "초 안에 받지 못했습니다.");
            }
            throw error;
        } finally {
            window.clearTimeout(timer);
            if (outerSignal) outerSignal.removeEventListener("abort", onAbort);
        }
    }

    async function responseMessage(response, fallback) {
        try {
            const data = await response.json();
            return data.detail || fallback;
        } catch (_) {
            return fallback;
        }
    }

    async function createJobWithRetry(form, requestId, signal) {
        let lastError = null;
        for (let attempt = 0; attempt < 3; attempt += 1) {
            let response = null;
            let responseData = {};
            try {
                const result = await fetchJsonWithTimeout(
                    "/api/jobs",
                    { method: "POST", body: form },
                    signal,
                    5 * 60 * 1000
                );
                response = result.response;
                responseData = result.data;
            } catch (error) {
                if (error && error.name === "AbortError") throw error;
                lastError = error;
            }
            if (response) {
                if (response.ok) {
                    return responseData;
                } else if (!retryableStatus(response.status)) {
                    throw new Error(responseData.detail || "변환을 시작하지 못했습니다.");
                }
            }

            // POST 응답만 유실됐을 수 있으므로, 새 작업을 만들기 전에 이미 같은
            // 요청 번호로 시작됐는지 확인한다.
            let statusResponse = null;
            let statusData = {};
            try {
                const result = await fetchJsonWithTimeout(
                    "/api/jobs/" + requestId,
                    { cache: "no-store" },
                    signal
                );
                statusResponse = result.response;
                statusData = result.data;
            } catch (error) {
                if (error && error.name === "AbortError") throw error;
                lastError = error;
            }
            if (statusResponse) {
                if (statusResponse.ok) {
                    return statusData;
                } else if (statusResponse.status === 401 || statusResponse.status === 403) {
                    throw new Error(statusData.detail || "로그인이 만료되었습니다.");
                }
            }
            await wait(500 * (attempt + 1), signal);
        }

        const error = new Error(
            (lastError && lastError.message ? lastError.message + " " : "") +
            "서버 응답을 확인하지 못했습니다. 연결을 확인한 뒤 변환 시작을 다시 누르면 같은 요청을 이어서 확인합니다."
        );
        error.requestUncertain = true;
        throw error;
    }

    // ── 글자를 읽을 곳 ───────────────────────────────────────────────────
    function selectedProcessor() {
        if (browserSupported && el.processorBrowser.checked) return "browser";
        return "server";
    }

    function applyProcessorModeRule() {
        const onBrowser = selectedProcessor() === "browser";
        if (onBrowser) {
            // 브라우저 조립은 아직 원본 유지 방식만 지원한다.
            el.modeKeep.checked = true;
        }
        el.dpi.disabled = onBrowser;
        el.dpiDescription.textContent = onBrowser
            ? "그래픽카드 방식에서는 글자 인식에 알맞은 크기로 자동 조정합니다."
            : "숫자가 클수록 작은 글씨까지 잘 읽지만 더 오래 걸립니다.";
        el.modeRebuild.disabled = onBrowser;
        el.modeRebuildChoice.classList.toggle("is-disabled", onBrowser);
        el.modeRebuildChoice.setAttribute("aria-disabled", String(onBrowser));
        applyCorrectionRule();
    }

    // ── 비뚤어진 스캔본 바로잡기 규칙 ────────────────────────────────────
    function applyCorrectionRule() {
        const onBrowser = selectedProcessor() === "browser";
        // 미세 기울기 보정은 페이지를 다시 그려야 하므로 '용량 줄이기'에서만 쓸 수 있다.
        const canDeskew = !onBrowser && el.modeRebuild.checked;

        el.autoRotate.disabled = onBrowser;
        el.autoRotateChoice.classList.toggle("is-disabled", onBrowser);
        el.autoRotateChoice.setAttribute("aria-disabled", String(onBrowser));

        el.deskew.disabled = !canDeskew;
        el.deskewChoice.classList.toggle("is-disabled", !canDeskew);
        el.deskewChoice.setAttribute("aria-disabled", String(!canDeskew));
        if (!canDeskew) el.deskew.checked = false;

        if (onBrowser) {
            el.correctionNote.textContent =
                "그래픽카드 방식에서는 방향·기울기 보정을 쓸 수 없습니다. " +
                "쓰시려면 ‘서버에서 읽기’를 골라 주세요.";
        } else if (!canDeskew) {
            el.correctionNote.textContent =
                "‘살짝 기운 것까지 펴기’는 ‘용량 줄이기’를 골라야 켤 수 있습니다. " +
                "원본을 그대로 두면서 픽셀을 다시 그릴 수는 없기 때문입니다.";
        } else {
            el.correctionNote.textContent =
                "기운 쪽만 골라서 다시 그립니다. 곧게 들어온 쪽은 그대로 둡니다.";
        }
    }

    function selectedExportFormats() {
        return Array.from(el.exportFormats)
            .filter((box) => box.checked)
            .map((box) => box.value);
    }

    function checkBrowserSupport() {
        browserSupported = window.isSecureContext && Boolean(navigator.gpu);
        el.processorBrowser.disabled = !browserSupported;
        el.processorBrowserChoice.classList.toggle("is-disabled", !browserSupported);
        el.processorBrowserChoice.setAttribute("aria-disabled", String(!browserSupported));

        if (browserSupported) {
            el.browserProcessorHint.textContent =
                "이 브라우저에서는 그래픽카드 시험 기능을 선택할 수 있습니다.";
        } else {
            el.processorServer.checked = true;
            if (!window.isSecureContext) {
                el.browserProcessorHint.textContent =
                    "그래픽카드 방식은 안전한 HTTPS 주소에서만 쓸 수 있어 서버 방식을 선택했습니다.";
            } else {
                el.browserProcessorHint.textContent =
                    "이 브라우저는 그래픽카드 방식을 지원하지 않아 서버 방식을 선택했습니다. " +
                    "최신 Chrome 또는 Edge에서 다시 확인해 주세요.";
            }
        }
        applyProcessorModeRule();
    }

    el.processorServer.addEventListener("change", applyProcessorModeRule);
    el.processorBrowser.addEventListener("change", applyProcessorModeRule);
    el.modeKeep.addEventListener("change", applyCorrectionRule);
    el.modeRebuild.addEventListener("change", applyCorrectionRule);

    // ── 고른 파일 표시 ────────────────────────────────────────────────────
    function markReady(label) {
        el.pickedFile.textContent = label;
        el.startButton.disabled = false;
        el.readyHint.textContent = "이제 변환을 시작할 수 있습니다.";
    }

    // 고른 문서를 가운데 화면에 띄운다. 보기에 실패해도 변환은 할 수 있으므로
    // 오류로 처리하지 않고 그대로 진행한다.
    function preview(source) {
        if (!window.PdfViewer) return;
        window.PdfViewer.open(source).catch(() => {});
    }

    function pickFile(file) {
        if (!file) return;
        if (!file.name.toLowerCase().endsWith(".pdf")) {
            showError("PDF 파일만 올릴 수 있습니다. 고른 파일: " + file.name);
            return;
        }
        clearError();
        pickedFile = file;
        pickedStored = null;   // 두 곳에서 동시에 고를 수 없다
        markReady(file.name + " · " + formatSize(file.size));
        preview(file);
    }

    function pickStored(item, node) {
        clearError();
        pickedStored = item;
        pickedFile = null;
        el.fileInput.value = "";
        el.storageList.querySelectorAll(".storage-item").forEach((row) => {
            row.classList.toggle("is-picked", row === node);
        });
        markReady(item.relative + " · " + formatSize(item.size));
        preview("/api/storage/file?relative=" + encodeURIComponent(item.relative));
    }

    // ── 왼쪽 도구 줄 ──────────────────────────────────────────────────────
    // 도구를 고르면 오른쪽 패널에 그 설정만 보여 준다. 설정을 모두 늘어놓으면
    // 화면이 길어져서, 지금 하려는 일과 상관없는 것까지 지나쳐야 한다.
    const RAIL = [
        { button: el.railFile, tool: el.toolFile },
        { button: el.railOcr, tool: el.toolOcr },
        { button: el.railFix, tool: el.toolFix },
        { button: el.railExport, tool: el.toolExport },
    ];

    function switchTool(picked) {
        RAIL.forEach(({ button, tool }) => {
            const isPicked = button === picked;
            button.classList.toggle("is-active", isPicked);
            button.setAttribute("aria-selected", String(isPicked));
            tool.hidden = !isPicked;
            tool.classList.toggle("is-active", isPicked);
        });
    }

    RAIL.forEach(({ button }) => {
        button.addEventListener("click", () => switchTool(button));
    });

    el.filePick.addEventListener("click", () => el.fileInput.click());

    // ── 탭 바꾸기 ─────────────────────────────────────────────────────────
    function switchTab(name) {
        const onUpload = name === "upload";
        el.tabUploadButton.classList.toggle("is-active", onUpload);
        el.tabStorageButton.classList.toggle("is-active", !onUpload);
        el.tabUploadButton.setAttribute("aria-selected", String(onUpload));
        el.tabStorageButton.setAttribute("aria-selected", String(!onUpload));
        el.tabUploadButton.tabIndex = onUpload ? 0 : -1;
        el.tabStorageButton.tabIndex = onUpload ? -1 : 0;
        el.tabUpload.classList.toggle("is-hidden", !onUpload);
        el.tabStorage.classList.toggle("is-hidden", onUpload);

        if (!onUpload && el.storageList.children.length === 0) {
            loadStorage();
        }
    }

    el.tabUploadButton.addEventListener("click", () => switchTab("upload"));
    el.tabStorageButton.addEventListener("click", () => switchTab("storage"));
    [el.tabUploadButton, el.tabStorageButton].forEach((tab, index, tabs) => {
        tab.addEventListener("keydown", (event) => {
            let nextIndex = null;
            if (event.key === "ArrowRight" || event.key === "ArrowDown") {
                nextIndex = (index + 1) % tabs.length;
            } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
                nextIndex = (index - 1 + tabs.length) % tabs.length;
            } else if (event.key === "Home") {
                nextIndex = 0;
            } else if (event.key === "End") {
                nextIndex = tabs.length - 1;
            }
            if (nextIndex === null) return;

            event.preventDefault();
            switchTab(nextIndex === 0 ? "upload" : "storage");
            tabs[nextIndex].focus();
        });
    });

    // ── 보관함 목록 ───────────────────────────────────────────────────────
    async function loadStorage() {
        // 불러오는 중에는 같은 크기의 자리표시를 보여 준다. (빈 화면은 고장으로 보인다)
        el.storageHint.textContent = "보관함을 불러오는 중입니다.";
        el.storageList.innerHTML = "";
        for (let i = 0; i < 3; i += 1) {
            const skeleton = document.createElement("div");
            skeleton.className = "storage-skeleton";
            el.storageList.appendChild(skeleton);
        }

        try {
            const res = await fetch("/api/storage/files");
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || "보관함을 읽지 못했습니다.");
            }

            el.storageList.innerHTML = "";
            if (data.files.length === 0) {
                el.storageHint.textContent =
                    "보관함이 비어 있습니다. 서버의 '" + data.source_dir + "' 폴더에 PDF를 넣어 주세요.";
                return;
            }

            el.storageHint.textContent =
                "'" + data.source_dir + "' 안의 PDF " + data.files.length + "개입니다.";

            data.files.forEach((item) => {
                const row = document.createElement("label");
                row.className = "storage-item";

                const radio = document.createElement("input");
                radio.type = "radio";
                radio.name = "stored-file";

                const body = document.createElement("span");
                body.className = "storage-item-body";
                const name = document.createElement("span");
                name.className = "storage-item-name";
                name.textContent = item.relative;
                const meta = document.createElement("span");
                meta.className = "storage-item-meta";
                meta.textContent = formatSize(item.size) + " · " + item.modified;
                body.append(name, meta);

                row.append(radio, body);
                row.addEventListener("click", () => {
                    radio.checked = true;
                    pickStored(item, row);
                });
                el.storageList.appendChild(row);
            });
        } catch (err) {
            el.storageList.innerHTML = "";
            el.storageHint.textContent = err.message;
        }
    }

    el.storageRefresh.addEventListener("click", loadStorage);

    // ── 로그아웃 ──────────────────────────────────────────────────────────
    el.logoutButton.addEventListener("click", async () => {
        const cleanupJobId = currentJobId;
        if (cleanupJobId && (
            pendingJobRequest || !el.stopButton.classList.contains("is-hidden")
        )) {
            abortCreateRequest();
            abortBrowserWork();
            try {
                const stopResponse = await fetchWithTimeout(
                    "/api/jobs/" + cleanupJobId + "/stop",
                    { method: "POST" },
                    null,
                    5000
                );
                if (!stopResponse.ok) throw new Error("작업 정리 요청 실패");
            } catch (_) {
                // 첫 요청의 응답만 유실됐을 수 있어, 세션을 지우기 전에 한 번 더 맡긴다.
                stopJobInBackground(cleanupJobId);
            }
        }
        const res = await fetch("/api/logout", { method: "POST" });
        const data = await res.json();
        window.location.href = data.next || "/login";
    });

    el.dropzone.addEventListener("click", () => el.fileInput.click());
    el.dropzone.addEventListener("keydown", (event) => {
        // 마우스 없이 키보드만으로도 파일을 고를 수 있어야 한다.
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            el.fileInput.click();
        }
    });
    el.fileInput.addEventListener("change", () => pickFile(el.fileInput.files[0]));

    // 끌어다 놓기는 문서 보는 칸 전체에서 받는다. 문서를 이미 열어 둔 뒤에도
    // 다른 파일로 바꿀 수 있어야 하기 때문이다.
    ["dragenter", "dragover"].forEach((name) => {
        el.stageView.addEventListener(name, (event) => {
            event.preventDefault();
            el.stageView.classList.add("is-over");
            el.dropzone.classList.add("is-over");
        });
    });
    ["dragleave", "drop"].forEach((name) => {
        el.stageView.addEventListener(name, (event) => {
            event.preventDefault();
            el.stageView.classList.remove("is-over");
            el.dropzone.classList.remove("is-over");
        });
    });
    el.stageView.addEventListener("drop", (event) => {
        pickFile(event.dataTransfer.files[0]);
    });

    async function runBrowserJob(job, score, controller) {
        try {
            const { runBrowserOcr } = await import("/static/browser_ocr.js");
            await runBrowserOcr({
                jobId: job.id,
                totalPages: job.total_pages,
                textScore: score,
                signal: controller.signal,
                onStage: (message) => {
                    if (currentJobId === job.id && !controller.signal.aborted) {
                        el.progressMessage.textContent = message;
                    }
                },
            });
        } catch (err) {
            if (controller.signal.aborted || (err && err.name === "AbortError")) return;

            // 브라우저 계산이 실패하면 서버에 남은 원본과 쪽 결과부터 정리한다.
            try {
                const stopResponse = await fetch("/api/jobs/" + job.id + "/stop", { method: "POST" });
                if (!stopResponse.ok) throw new Error("작업 정리 요청 실패");
            } catch (_) {
                // 아래 안내가 재시도 방법까지 알려 주므로 정리 요청 오류를 따로 덮어쓰지 않는다.
            }

            if (currentJobId !== job.id) return;
            const reason = err && err.message ? err.message + " " : "";
            showError(
                reason +
                "이 컴퓨터의 그래픽카드로 읽지 못했습니다. " +
                "고른 원본은 그대로 두었습니다. ‘서버에서 읽기’를 선택한 뒤 다시 시작해 주세요."
            );
            el.progressMessage.textContent = "브라우저 인식이 중지되었습니다. 서버 방식으로 다시 시도해 주세요.";
            el.startButton.disabled = false;
            hide(el.stopButton);
        } finally {
            if (browserController === controller) {
                browserController = null;
            }
        }
    }

    // ── 변환 시작 ─────────────────────────────────────────────────────────
    el.startButton.addEventListener("click", async () => {
        if (!pickedFile && !pickedStored) return;

        const score = parseFloat(el.textScore.value);
        if (!Number.isFinite(score) || score < 0 || score > 1) {
            showError("글자로 인정할 기준 점수는 0과 1 사이의 숫자여야 합니다. 예: 0.5");
            el.textScore.focus();
            return;
        }

        const processor = selectedProcessor();
        applyProcessorModeRule();

        clearError();
        hide(el.resultCard);
        hide(el.resetButton);
        hide(el.textLayerWarning);
        hide(el.exportError);
        closeProgressConnection();
        abortCreateRequest();
        abortBrowserWork();
        el.startButton.disabled = true;
        el.startButton.textContent = "변환을 요청하는 중입니다.";

        const form = new FormData();
        if (pickedFile) {
            form.append("file", pickedFile);
        } else {
            form.append("source_path", pickedStored.relative);
        }
        const exportFormats = selectedExportFormats();
        const autoRotate = !el.autoRotate.disabled && el.autoRotate.checked;
        const deskew = !el.deskew.disabled && el.deskew.checked;

        form.append("dpi", el.dpi.value);
        form.append("text_score", String(score));
        form.append("mode", document.querySelector('input[name="mode"]:checked').value);
        form.append("processor", processor);
        form.append("export_formats", exportFormats.join(","));
        form.append("auto_rotate", String(autoRotate));
        form.append("deskew", String(deskew));

        const requestSignature = JSON.stringify({
            source: pickedFile
                ? [pickedFile.name, pickedFile.size, pickedFile.lastModified]
                : [pickedStored.relative, pickedStored.size],
            dpi: el.dpi.value,
            score,
            mode: document.querySelector('input[name="mode"]:checked').value,
            processor,
            exportFormats,
            autoRotate,
            deskew,
        });
        if (pendingJobRequest && pendingJobRequest.signature !== requestSignature) {
            try {
                await cancelPendingJobRequest();
            } catch (error) {
                showError(
                    (error && error.message ? error.message + " " : "") +
                    "이전 요청을 확인한 뒤 다시 시작해 주세요. 중복 변환을 막기 위해 새 작업은 만들지 않았습니다."
                );
                el.startButton.disabled = false;
                el.startButton.textContent = "변환 시작";
                return;
            }
        }
        if (!pendingJobRequest) {
            pendingJobRequest = { id: newRequestId(), signature: requestSignature };
        }
        form.append("request_id", pendingJobRequest.id);

        const request = pendingJobRequest;
        const controller = new AbortController();
        createController = controller;
        currentJobId = request.id;
        currentProcessor = processor;
        show(el.statusBadge);
        setBadge(el.statusBadge, "실행 중");
        show(el.stopButton);
        el.stopButton.disabled = false;
        show(el.progressCard);
        setBadge(el.progressBadge, "실행 중");
        el.progressBar.style.width = "0%";
        el.progressPercent.textContent = "0%";
        el.progressMessage.textContent = "원본 PDF를 서버에 안전하게 전달하는 중입니다.";

        let job;
        try {
            job = await createJobWithRetry(form, request.id, controller.signal);
        } catch (err) {
            if (controller.signal.aborted || (err && err.name === "AbortError")) {
                // stop 요청의 서버 응답이 확인될 때까지 alias를 보존한다. 먼저
                // 지우면 정리 실패 뒤 같은 작업을 다시 찾을 방법이 없어진다.
                el.progressMessage.textContent = "서버가 중지 요청을 확인하는 중입니다.";
                return;
            }
            showError(err.message);
            if (!err.requestUncertain && pendingJobRequest === request) {
                pendingJobRequest = null;
                if (currentJobId === request.id) currentJobId = null;
                hide(el.stopButton);
            } else if (err.requestUncertain) {
                show(el.stopButton);
                el.stopButton.disabled = false;
            }
            el.startButton.disabled = false;
            el.startButton.textContent = "변환 시작";
            return;
        } finally {
            if (createController === controller) createController = null;
        }

        if (pendingJobRequest === request) pendingJobRequest = null;
        currentJobId = job.id;
        currentProcessor = job.processor || processor;

        if (job.has_text_layer) {
            // 스캐너가 자체 OCR로 만든 파일일 가능성이 높다. 그대로 한 번 더 얹으면
            // 글자가 두 겹이 되어 검색 결과가 엉킨다.
            el.textLayerWarning.textContent =
                "이 PDF에는 이미 글자가 들어 있습니다. 스캐너나 다른 프로그램에서 " +
                "글자 인식을 한 번 거친 파일로 보입니다. 이대로 변환하면 글자가 두 겹으로 " +
                "얹혀 검색 결과가 엉킬 수 있으니, 원하지 않으시면 ‘중지’를 눌러 주세요.";
            show(el.textLayerWarning);
        }
        el.startButton.textContent = "변환 시작";
        show(el.statusBadge);
        setBadge(el.statusBadge, "실행 중");
        show(el.stopButton);
        el.stopButton.disabled = false;

        // 진행 상황이 아직 오기 전이라도 빈 화면을 보여 주지 않는다.
        show(el.progressCard);
        el.progressBar.style.width = "0%";
        el.progressPercent.textContent = "0%";
        setBadge(el.progressBadge, "실행 중");
        el.progressMessage.textContent = currentProcessor === "browser"
            ? "이 컴퓨터의 그래픽카드 인식 도구를 준비하는 중입니다."
            : "전체 " + job.total_pages + "쪽입니다. 첫 쪽을 읽는 중입니다.";

        // 서버 진행 소식을 먼저 연결해야 첫 상태와 완료 신호를 놓치지 않는다.
        listenProgress(job.id);
        if (currentProcessor === "browser" && !job.finished) {
            const controller = new AbortController();
            browserController = controller;
            void runBrowserJob(job, score, controller);
        }
    });

    // ── 진행 상황 받기 ────────────────────────────────────────────────────
    function listenProgress(jobId) {
        const source = new EventSource("/api/jobs/" + jobId + "/events");
        eventSource = source;
        let progressRevision = 0;
        let greatestDonePages = -1;
        let recoveryInFlight = false;
        let terminalSeen = false;

        function applyProgress(data) {
            if (terminalSeen) return true;
            const donePages = Number(data.done_pages);
            if (!data.finished && Number.isFinite(donePages) && donePages < greatestDonePages) {
                return false;
            }
            if (Number.isFinite(donePages)) {
                greatestDonePages = Math.max(greatestDonePages, donePages);
            }
            progressRevision += 1;
            el.progressPercent.textContent = data.percent + "%";
            el.progressBar.style.width = data.percent + "%";
            el.progressMessage.textContent = data.message;
            setBadge(el.progressBadge, data.status);
            setBadge(el.statusBadge, data.status);

            if (!data.finished) return false;
            terminalSeen = true;

            source.close();
            if (eventSource === source) eventSource = null;
            abortBrowserWork();
            hide(el.stopButton);
            el.startButton.disabled = false;

            if (data.status === "성공") {
                clearError();
                showResult(data.result);
            } else if (data.status === "취소") {
                el.resultHint.textContent = "";
                hide(el.resultCard);
            } else {
                showError("변환이 실패했습니다. " + (data.error || ""));
            }
            return true;
        }

        source.onmessage = (event) => {
            if (currentJobId !== jobId) {
                source.close();
                return;
            }
            applyProgress(JSON.parse(event.data));
        };

        source.onerror = async () => {
            if (
                currentJobId !== jobId || eventSource !== source ||
                recoveryInFlight || terminalSeen
            ) return;
            recoveryInFlight = true;
            const startingRevision = progressRevision;
            // 연결이 끊긴 경우. 브라우저가 자동으로 다시 붙지만 사용자에게 알려 준다.
            el.progressMessage.textContent =
                "서버와 연결이 잠시 끊겼습니다. 다시 연결하는 중입니다.";

            try {
                const response = await fetch("/api/jobs/" + jobId, { cache: "no-store" });
                if (
                    currentJobId !== jobId || eventSource !== source ||
                    startingRevision !== progressRevision
                ) return;
                if (response.ok) {
                    const data = await response.json();
                    if (startingRevision !== progressRevision) return;
                    const finished = applyProgress(data);
                    if (!finished) {
                        el.progressMessage.textContent =
                            "서버와 연결이 잠시 끊겼습니다. 다시 연결하는 중입니다.";
                    }
                    return;
                }
                if (response.status === 401 || response.status === 403) {
                    source.close();
                    window.location.href = "/login";
                    return;
                }
                if (response.status !== 404) return;

                source.close();
                if (eventSource === source) eventSource = null;
                abortBrowserWork();
                hide(el.stopButton);
                el.startButton.disabled = false;
                showError(
                    "서버가 다시 시작되어 진행하던 작업 정보를 찾지 못했습니다. 원본을 골라 다시 시작해 주세요."
                );
            } catch (_) {
                // 인터넷 자체가 끊겼다면 EventSource의 기본 재연결을 그대로 기다린다.
            } finally {
                recoveryInFlight = false;
            }
        };
    }

    // ── 중지 ──────────────────────────────────────────────────────────────
    el.stopButton.addEventListener("click", async () => {
        if (!currentJobId) return;
        const stoppedJobId = currentJobId;
        const wasCreating = Boolean(createController) || Boolean(
            pendingJobRequest && pendingJobRequest.id === stoppedJobId
        );
        el.stopButton.disabled = true;
        el.stopButton.textContent = "중지하는 중";
        abortCreateRequest();
        if (currentProcessor === "browser") {
            // 이미지 받기·결과 보내기를 먼저 끊고, 서버 임시자료도 이어서 정리한다.
            abortBrowserWork();
        }
        try {
            const { response, data } = await fetchJsonWithTimeout(
                "/api/jobs/" + stoppedJobId + "/stop",
                { method: "POST" },
                null,
                5000
            );
            if (!response.ok) {
                throw new Error(data.detail || "중지 요청을 받아들이지 못했습니다.");
            }
            if (wasCreating) {
                if (pendingJobRequest && pendingJobRequest.id === stoppedJobId) {
                    pendingJobRequest = null;
                }
                if (currentJobId === stoppedJobId) currentJobId = null;
                el.progressMessage.textContent = "사용자 요청으로 중지했습니다.";
                setBadge(el.progressBadge, "취소");
                setBadge(el.statusBadge, "취소");
                hide(el.stopButton);
                el.startButton.disabled = false;
                el.startButton.textContent = "변환 시작";
            }
        } catch (err) {
            showError(err.message || "중지 요청을 보내지 못했습니다.");
            stopJobInBackground(stoppedJobId);
            el.stopButton.disabled = false;
        }
        el.stopButton.textContent = "중지";
    });

    // ── 결과 보여 주기 ────────────────────────────────────────────────────
    function showResult(result) {
        const sizeText = result.size_percent === 0 && result.output_bytes > 0
            ? "원본의 1% 미만으로 줄었습니다"
            : result.size_percent <= 100
            ? "원본의 " + result.size_percent + "%로 줄었습니다"
            : "원본의 " + result.size_percent + "%로 늘었습니다";

        const items = [
            { label: "걸린 시간", value: result.elapsed + "초",
              sub: "쪽당 " + (result.elapsed / Math.max(result.pages, 1)).toFixed(2) + "초" },
            { label: "처리한 쪽수", value: result.pages + "쪽",
              sub: "찾은 글줄 " + result.text_lines + "개" },
            { label: "변환 후 용량", value: formatSize(result.output_bytes), sub: sizeText },
            { label: "사용한 장치", value: DEVICE_LABEL[result.device] || result.device,
              sub: DEVICE_HINT[result.device] || "" },
        ];

        if (result.rotated_by || result.deskewed_pages) {
            items.push({
                label: "바로잡기",
                value: result.rotated_by ? result.rotated_by + "도 돌림" : "기울기만 폄",
                sub: result.deskewed_pages
                    ? "기운 쪽 " + result.deskewed_pages + "쪽을 폈습니다"
                    : "돌리는 각도만 바꿔 화질은 그대로입니다",
            });
        }

        el.resultGrid.innerHTML = "";
        items.forEach((item) => {
            const box = document.createElement("div");
            box.className = "result-item";
            const dt = document.createElement("dt");
            dt.textContent = item.label;
            const dd = document.createElement("dd");
            dd.textContent = item.value;
            const sub = document.createElement("span");
            sub.className = "sub";
            sub.textContent = item.sub;
            dd.appendChild(sub);
            box.append(dt, dd);
            el.resultGrid.appendChild(box);
        });

        // 함께 만든 글자 파일 내려받기 단추
        const EXPORT_LABEL = { txt: "글자 파일(txt)", md: "글자 파일(md)", docx: "워드 문서(docx)" };
        el.exportButtons.innerHTML = "";
        (result.exports || []).forEach((item) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "button-sub";
            button.textContent = (EXPORT_LABEL[item.format] || item.format) + " 내려받기";
            button.addEventListener("click", () => {
                if (!currentJobId) return;
                window.location.href =
                    "/api/jobs/" + currentJobId + "/download/" + item.format;
            });
            el.exportButtons.appendChild(button);
        });

        if (result.export_error) {
            el.exportError.textContent =
                "PDF는 무사히 만들었지만 글자 파일은 저장하지 못했습니다. (" +
                result.export_error + ")";
            show(el.exportError);
        } else {
            hide(el.exportError);
        }

        const savedText = (result.exports || []).length
            ? " 읽어 낸 글자는 " +
              result.exports.map((item) => item.name).join(", ") + " 로도 저장했습니다."
            : "";
        el.resultHint.textContent =
            "이제 PDF 안에서 글자를 찾을 수 있습니다. " +
            "서버의 '" + result.saved_dir + "' 폴더에 " + result.saved_name +
            " 으로 저장했습니다." + savedText;
        show(el.resultCard);
        show(el.resetButton);

        // 만들어진 결과를 그 자리에서 바로 보여 준다. 내려받아 다른 프로그램으로
        // 열어 보지 않아도 방향이 바로 섰는지, 글자가 얹혔는지 확인할 수 있다.
        if (currentJobId) {
            preview("/api/jobs/" + currentJobId + "/download");
            el.pickedFile.textContent = result.saved_name + " (변환 결과)";
        }
    }

    el.downloadButton.addEventListener("click", () => {
        if (!currentJobId) return;
        window.location.href = "/api/jobs/" + currentJobId + "/download";
    });

    // ── 처음 상태로 되돌리기 ──────────────────────────────────────────────
    el.resetButton.addEventListener("click", () => {
        discardPendingJobRequest();
        abortCreateRequest();
        abortBrowserWork();
        closeProgressConnection();
        pickedFile = null;
        pickedStored = null;
        currentJobId = null;
        currentProcessor = "server";
        el.fileInput.value = "";
        el.storageList.querySelectorAll(".storage-item").forEach((row) => {
            row.classList.remove("is-picked");
        });
        el.pickedFile.textContent = "";
        el.startButton.disabled = true;
        el.readyHint.textContent = "파일을 고르면 시작할 수 있습니다.";
        hide(el.statusBadge);
        hide(el.stopButton);
        hide(el.resetButton);
        hide(el.progressCard);
        hide(el.resultCard);
        hide(el.textLayerWarning);
        hide(el.exportError);
        clearError();
        if (window.PdfViewer) window.PdfViewer.close().catch(() => {});
        switchTool(el.railFile);
        el.dropzone.focus();
    });

    window.addEventListener("pagehide", () => {
        if (
            currentJobId &&
            (pendingJobRequest || (
                currentProcessor === "browser" &&
                !el.stopButton.classList.contains("is-hidden")
            ))
        ) {
            const cleanupJobId = currentJobId;
            abortCreateRequest();
            abortBrowserWork();
            stopJobInBackground(cleanupJobId);
        }
    });

    checkBrowserSupport();
})();
