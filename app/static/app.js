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

    let pickedFile = null;
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

    // ── 파일 고르기 ───────────────────────────────────────────────────────
    function pickFile(file) {
        if (!file) return;
        if (!file.name.toLowerCase().endsWith(".pdf")) {
            showError("PDF 파일만 올릴 수 있습니다. 고른 파일: " + file.name);
            return;
        }
        clearError();
        pickedFile = file;
        el.pickedFile.textContent = file.name + " (" + formatSize(file.size) + ")";
        el.pickedFile.classList.add("is-shown");
        el.startButton.disabled = false;
        el.readyHint.textContent = "이제 변환을 시작할 수 있습니다.";
    }

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
        if (!pickedFile) return;

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
        form.append("file", pickedFile);
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
            { label: "사용한 장치", value: result.device === "GPU" ? "그래픽카드" : "중앙처리장치",
              sub: result.device === "GPU" ? "GPU 가속 사용" : "GPU를 쓰지 못해 CPU로 처리" },
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

        el.resultHint.textContent = "이제 PDF 안에서 글자를 찾을 수 있습니다.";
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
        currentJobId = null;
        el.fileInput.value = "";
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
