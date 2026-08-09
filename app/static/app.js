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
        pickedFile: document.getElementById("picked-file"),
        logoutButton: document.getElementById("logout-button"),
        tabUploadButton: document.getElementById("tab-upload-button"),
        tabStorageButton: document.getElementById("tab-storage-button"),
        tabUpload: document.getElementById("tab-upload"),
        tabStorage: document.getElementById("tab-storage"),
        storageList: document.getElementById("storage-list"),
        storageHint: document.getElementById("storage-hint"),
        storageRefresh: document.getElementById("storage-refresh"),
        dpi: document.getElementById("dpi"),
        textScore: document.getElementById("text-score"),
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
    };
    const DEVICE_HINT = {
        "CUDA": "NVIDIA 가속 사용",
        "DirectML": "DirectML 가속 사용",
        "CPU": "그래픽 가속을 쓰지 못해 CPU로 처리",
    };

    let pickedFile = null;      // 내 컴퓨터에서 고른 파일
    let pickedStored = null;    // 보관함에서 고른 파일 {relative, name, size}
    let currentJobId = null;
    let eventSource = null;

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

    // ── 고른 파일 표시 ────────────────────────────────────────────────────
    function markReady(label) {
        el.pickedFile.textContent = label;
        el.pickedFile.classList.add("is-shown");
        el.startButton.disabled = false;
        el.readyHint.textContent = "이제 변환을 시작할 수 있습니다.";
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
        markReady("올릴 파일: " + file.name + " (" + formatSize(file.size) + ")");
    }

    function pickStored(item, node) {
        clearError();
        pickedStored = item;
        pickedFile = null;
        el.fileInput.value = "";
        el.storageList.querySelectorAll(".storage-item").forEach((row) => {
            row.classList.toggle("is-picked", row === node);
        });
        markReady("보관함 파일: " + item.relative + " (" + formatSize(item.size) + ")");
    }

    // ── 탭 바꾸기 ─────────────────────────────────────────────────────────
    function switchTab(name) {
        const onUpload = name === "upload";
        el.tabUploadButton.classList.toggle("is-active", onUpload);
        el.tabStorageButton.classList.toggle("is-active", !onUpload);
        el.tabUploadButton.setAttribute("aria-selected", String(onUpload));
        el.tabStorageButton.setAttribute("aria-selected", String(!onUpload));
        el.tabUpload.classList.toggle("is-hidden", !onUpload);
        el.tabStorage.classList.toggle("is-hidden", onUpload);

        if (!onUpload && el.storageList.children.length === 0) {
            loadStorage();
        }
    }

    el.tabUploadButton.addEventListener("click", () => switchTab("upload"));
    el.tabStorageButton.addEventListener("click", () => switchTab("storage"));

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

    ["dragenter", "dragover"].forEach((name) => {
        el.dropzone.addEventListener(name, (event) => {
            event.preventDefault();
            el.dropzone.classList.add("is-over");
        });
    });
    ["dragleave", "drop"].forEach((name) => {
        el.dropzone.addEventListener(name, (event) => {
            event.preventDefault();
            el.dropzone.classList.remove("is-over");
        });
    });
    el.dropzone.addEventListener("drop", (event) => {
        pickFile(event.dataTransfer.files[0]);
    });

    // ── 변환 시작 ─────────────────────────────────────────────────────────
    el.startButton.addEventListener("click", async () => {
        if (!pickedFile && !pickedStored) return;

        const score = parseFloat(el.textScore.value);
        if (Number.isNaN(score) || score < 0 || score > 1) {
            showError("글자로 인정할 기준 점수는 0과 1 사이의 숫자여야 합니다. 예: 0.5");
            el.textScore.focus();
            return;
        }

        clearError();
        hide(el.resultCard);
        el.startButton.disabled = true;
        el.startButton.textContent = "변환을 요청하는 중입니다.";

        const form = new FormData();
        if (pickedFile) {
            form.append("file", pickedFile);
        } else {
            form.append("source_path", pickedStored.relative);
        }
        form.append("dpi", el.dpi.value);
        form.append("text_score", String(score));
        form.append("mode", document.querySelector('input[name="mode"]:checked').value);

        let job;
        try {
            const res = await fetch("/api/jobs", { method: "POST", body: form });
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || "변환을 시작하지 못했습니다.");
            }
            job = data;
        } catch (err) {
            showError(err.message);
            el.startButton.disabled = false;
            el.startButton.textContent = "변환 시작";
            return;
        }

        currentJobId = job.id;
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
        el.progressMessage.textContent =
            "전체 " + job.total_pages + "쪽입니다. 첫 쪽을 읽는 중입니다.";

        listenProgress(job.id);
    });

    // ── 진행 상황 받기 ────────────────────────────────────────────────────
    function listenProgress(jobId) {
        eventSource = new EventSource("/api/jobs/" + jobId + "/events");

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            el.progressPercent.textContent = data.percent + "%";
            el.progressBar.style.width = data.percent + "%";
            el.progressMessage.textContent = data.message;
            setBadge(el.progressBadge, data.status);
            setBadge(el.statusBadge, data.status);

            if (!data.finished) return;

            eventSource.close();
            eventSource = null;
            hide(el.stopButton);
            el.startButton.disabled = false;

            if (data.status === "성공") {
                showResult(data.result);
            } else if (data.status === "취소") {
                el.resultHint.textContent = "";
                hide(el.resultCard);
            } else {
                showError("변환이 실패했습니다. " + (data.error || ""));
            }
        };

        eventSource.onerror = () => {
            // 연결이 끊긴 경우. 브라우저가 자동으로 다시 붙지만 사용자에게 알려 준다.
            el.progressMessage.textContent =
                "서버와 연결이 잠시 끊겼습니다. 다시 연결하는 중입니다.";
        };
    }

    // ── 중지 ──────────────────────────────────────────────────────────────
    el.stopButton.addEventListener("click", async () => {
        if (!currentJobId) return;
        el.stopButton.disabled = true;
        el.stopButton.textContent = "중지하는 중";
        try {
            await fetch("/api/jobs/" + currentJobId + "/stop", { method: "POST" });
        } catch (err) {
            showError("중지 요청을 보내지 못했습니다.");
            el.stopButton.disabled = false;
        }
        el.stopButton.textContent = "중지";
    });

    // ── 결과 보여 주기 ────────────────────────────────────────────────────
    function showResult(result) {
        const sizeText = result.size_percent <= 100
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

        el.resultHint.textContent =
            "이제 PDF 안에서 글자를 찾을 수 있습니다. " +
            "서버의 '" + result.saved_dir + "' 폴더에 " + result.saved_name + " 으로 저장했습니다.";
        show(el.resultCard);
    }

    el.downloadButton.addEventListener("click", () => {
        if (!currentJobId) return;
        window.location.href = "/api/jobs/" + currentJobId + "/download";
    });

    // ── 처음 상태로 되돌리기 ──────────────────────────────────────────────
    el.resetButton.addEventListener("click", () => {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        pickedFile = null;
        pickedStored = null;
        currentJobId = null;
        el.fileInput.value = "";
        el.storageList.querySelectorAll(".storage-item").forEach((row) => {
            row.classList.remove("is-picked");
        });
        el.pickedFile.textContent = "";
        el.pickedFile.classList.remove("is-shown");
        el.startButton.disabled = true;
        el.readyHint.textContent = "파일을 고르면 시작할 수 있습니다.";
        hide(el.statusBadge);
        hide(el.stopButton);
        hide(el.progressCard);
        hide(el.resultCard);
        clearError();
        el.dropzone.focus();
    });
})();
