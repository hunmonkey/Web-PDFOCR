// =============================================================================
// 문서 보기
//
// pdf.js로 PDF를 화면에 그린다. 왼쪽에는 쪽 목록(썸네일), 가운데에는 고른 쪽을
// 크게 보여 준다.
//
// 올린 파일은 브라우저가 직접 읽는다. 미리 보자고 서버에 올렸다 받는 왕복을
// 하지 않으므로 서버에 부담이 없고 화면도 곧바로 뜬다.
//
// app.js는 모듈이 아니라서 import로 주고받을 수 없다. 그래서 필요한 기능만
// window.PdfViewer 에 얹어 둔다.
// =============================================================================

import * as pdfjs from "/static/vendor/pdf.min.mjs";

// 글자를 풀어내는 무거운 일은 다른 갈래(worker)에서 돌린다. 이것을 지정하지 않으면
// 화면을 그리는 동안 브라우저 전체가 멈춘 것처럼 굳는다.
pdfjs.GlobalWorkerOptions.workerSrc = "/static/vendor/pdf.worker.min.mjs";

const el = {
    thumbsEmpty: document.getElementById("thumbs-empty"),
    thumbsList: document.getElementById("thumbs-list"),
    dropzone: document.getElementById("dropzone"),
    paper: document.getElementById("paper"),
    canvas: document.getElementById("page-canvas"),
    stageView: document.getElementById("stage-view"),
    stageNote: document.getElementById("stage-note"),
    pagePrev: document.getElementById("page-prev"),
    pageNext: document.getElementById("page-next"),
    pageInput: document.getElementById("page-input"),
    pageTotal: document.getElementById("page-total"),
    zoomIn: document.getElementById("zoom-in"),
    zoomOut: document.getElementById("zoom-out"),
    zoomValue: document.getElementById("zoom-value"),
    zoomFit: document.getElementById("zoom-fit"),
};

// 고를 수 있는 확대 단계. 사이를 너무 잘게 나누면 누를 때마다 거의 그대로라 답답하다.
const ZOOM_STEPS = [0.5, 0.75, 1, 1.25, 1.5, 2, 3];
const THUMB_WIDTH = 116;
// 아주 큰 화면에서 지나치게 큰 그림을 만들지 않도록 한계를 둔다.
const MAX_RENDER_SCALE = 4;

let document_ = null;      // 지금 열려 있는 PDF
// 문서를 닫는 destroy()는 문서 자체가 아니라 '여는 작업' 쪽에 있다. 그래서 함께 들고 있는다.
let loadingTask = null;
let pageCount = 0;
let currentPage = 1;
let zoom = "fit";          // 'fit' 이거나 배율 숫자
let lastScale = 1;         // 마지막으로 실제로 그린 배율. '맞춤'에서 확대할 때의 기준이 된다.
let renderTask = null;     // 그리는 중인 작업. 새 요청이 오면 취소한다.
let thumbObserver = null;
let openToken = 0;         // 늦게 도착한 옛 요청의 결과를 버리기 위한 표식

function show(node) { node.classList.remove("is-hidden"); }
function hide(node) { node.classList.add("is-hidden"); }

function setControlsEnabled(enabled) {
    for (const node of [el.pagePrev, el.pageNext, el.pageInput,
                        el.zoomIn, el.zoomOut, el.zoomFit]) {
        node.disabled = !enabled;
    }
}

function note(message) {
    if (message) {
        el.stageNote.textContent = message;
        show(el.stageNote);
    } else {
        hide(el.stageNote);
    }
}

// ── 확대 배율 계산 ──────────────────────────────────────────────────────────
function fitScale(page) {
    /** 문서 한 쪽 전체가 보이는 칸에 들어가는 배율을 구한다.
     *
     *  가로만 맞추면 세로로 긴 쪽은 아래가 잘려서, 한 쪽을 보려고 또 스크롤해야 한다.
     *  그래서 가로와 세로 중 더 빡빡한 쪽에 맞춘다.
     */
    const view = page.getViewport({ scale: 1 });
    const styles = getComputedStyle(el.stageView);
    const padX = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight);
    const padY = parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom);
    const roomX = el.stageView.clientWidth - padX;
    const roomY = el.stageView.clientHeight - padY;
    if (roomX <= 0 || roomY <= 0) return 1;
    return Math.min(roomX / view.width, roomY / view.height, MAX_RENDER_SCALE);
}

function currentScale(page) {
    return zoom === "fit" ? fitScale(page) : zoom;
}

function updateZoomLabel(scale) {
    lastScale = scale;
    el.zoomValue.textContent = zoom === "fit"
        ? "맞춤"
        : Math.round(scale * 100) + "%";
}

// ── 쪽 그리기 ───────────────────────────────────────────────────────────────
async function drawPage(number) {
    if (!document_) return;

    // 앞서 그리던 것이 있으면 멈춘다. 빠르게 쪽을 넘길 때 옛 그림이 나중에
    // 도착해 새 그림을 덮어쓰는 일을 막는다.
    if (renderTask) {
        renderTask.cancel();
        renderTask = null;
    }

    const token = openToken;
    const page = await document_.getPage(number);
    if (token !== openToken) return;

    // 화면이 촘촘한 기기에서도 글자가 또렷하도록 실제 점 개수에 맞춰 그린다.
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const scale = currentScale(page);
    const viewport = page.getViewport({ scale: scale * ratio });

    el.canvas.width = Math.floor(viewport.width);
    el.canvas.height = Math.floor(viewport.height);
    el.canvas.style.width = Math.floor(viewport.width / ratio) + "px";
    el.canvas.style.height = Math.floor(viewport.height / ratio) + "px";

    const task = page.render({
        canvas: el.canvas,
        canvasContext: el.canvas.getContext("2d"),
        viewport,
    });
    renderTask = task;

    try {
        await task.promise;
    } catch (error) {
        if (error && error.name === "RenderingCancelledException") return;
        throw error;
    } finally {
        if (renderTask === task) renderTask = null;
    }

    updateZoomLabel(scale);
    markCurrentThumb();
}

function markCurrentThumb() {
    for (const node of el.thumbsList.children) {
        const isCurrent = Number(node.dataset.page) === currentPage;
        node.classList.toggle("is-current", isCurrent);
        if (isCurrent) node.setAttribute("aria-current", "true");
        else node.removeAttribute("aria-current");
    }
}

async function goTo(number) {
    if (!document_) return;
    const wanted = Math.min(Math.max(Math.round(number) || 1, 1), pageCount);
    currentPage = wanted;
    el.pageInput.value = String(wanted);
    el.pagePrev.disabled = wanted <= 1;
    el.pageNext.disabled = wanted >= pageCount;
    await drawPage(wanted);
}

// ── 쪽 목록 ─────────────────────────────────────────────────────────────────
function buildThumbs() {
    /** 쪽 수만큼 자리를 먼저 만들어 두고, 화면에 들어온 것만 실제로 그린다.
     *  쪽수가 많은 문서에서 처음부터 전부 그리면 한참 멈춘다. */
    el.thumbsList.innerHTML = "";
    if (thumbObserver) thumbObserver.disconnect();

    thumbObserver = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            thumbObserver.unobserve(entry.target);
            void drawThumb(entry.target);
        }
    }, { root: document.getElementById("thumbs"), rootMargin: "200px" });

    for (let number = 1; number <= pageCount; number += 1) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "thumb";
        button.dataset.page = String(number);
        button.setAttribute("aria-label", number + "쪽 보기");

        const hold = document.createElement("div");
        hold.className = "thumb-hold";
        const label = document.createElement("span");
        label.className = "thumb-label";
        label.textContent = String(number);

        button.append(hold, label);
        button.addEventListener("click", () => { void goTo(number); });
        el.thumbsList.appendChild(button);
        thumbObserver.observe(button);
    }
    markCurrentThumb();
}

async function drawThumb(button) {
    const token = openToken;
    const number = Number(button.dataset.page);
    let page;
    try {
        page = await document_.getPage(number);
    } catch {
        return;
    }
    if (token !== openToken || !button.isConnected) return;

    const base = page.getViewport({ scale: 1 });
    const viewport = page.getViewport({ scale: THUMB_WIDTH / base.width });
    const canvas = document.createElement("canvas");
    canvas.width = Math.floor(viewport.width);
    canvas.height = Math.floor(viewport.height);

    try {
        await page.render({
            canvas,
            canvasContext: canvas.getContext("2d"),
            viewport,
        }).promise;
    } catch {
        return;
    }
    if (token !== openToken || !button.isConnected) return;

    const hold = button.querySelector(".thumb-hold");
    if (hold) hold.replaceWith(canvas);
}

// ── 열기와 닫기 ─────────────────────────────────────────────────────────────
async function toSource(input) {
    /** File, 주소 문자열, 이미 읽어 둔 바이트를 pdf.js가 아는 형태로 바꾼다. */
    if (typeof input === "string") return { url: input };
    if (input instanceof Blob) return { data: new Uint8Array(await input.arrayBuffer()) };
    return { data: input };
}

async function open(input) {
    await close();
    openToken += 1;
    const token = openToken;

    note("문서를 여는 중입니다.");
    const task = pdfjs.getDocument({
        ...(await toSource(input)),
        // 한글·한자 글꼴을 풀어내는 데 쓰는 대응표. 이것이 없으면 변환 결과처럼
        // 한국어 글꼴이 들어간 PDF에서 글꼴 경고가 나고 글자를 다룰 수 없다.
        cMapUrl: "/static/vendor/cmaps/",
        cMapPacked: true,
    });
    let opened;
    try {
        opened = await task.promise;
    } catch {
        if (token !== openToken) return false;
        note("이 파일은 화면에서 열어 볼 수 없습니다. 변환에는 문제가 없을 수 있습니다.");
        return false;
    }
    if (token !== openToken) {
        void task.destroy();
        return false;
    }

    loadingTask = task;
    document_ = opened;
    pageCount = opened.numPages;
    currentPage = 1;
    zoom = "fit";

    hide(el.dropzone);
    show(el.paper);
    note("");
    hide(el.thumbsEmpty);
    el.pageTotal.textContent = "/ " + pageCount;
    el.pageInput.max = String(pageCount);
    setControlsEnabled(true);

    buildThumbs();
    await goTo(1);
    return true;
}

async function close() {
    openToken += 1;
    if (renderTask) {
        renderTask.cancel();
        renderTask = null;
    }
    if (thumbObserver) {
        thumbObserver.disconnect();
        thumbObserver = null;
    }
    document_ = null;
    if (loadingTask) {
        const closing = loadingTask;
        loadingTask = null;
        await closing.destroy().catch(() => {});
    }

    pageCount = 0;
    currentPage = 1;
    zoom = "fit";
    el.thumbsList.innerHTML = "";
    show(el.thumbsEmpty);
    hide(el.paper);
    show(el.dropzone);
    note("");
    el.pageTotal.textContent = "/ 0";
    el.pageInput.value = "1";
    el.zoomValue.textContent = "맞춤";
    setControlsEnabled(false);
    el.canvas.width = 0;
    el.canvas.height = 0;
}

// ── 단추 동작 ───────────────────────────────────────────────────────────────
el.pagePrev.addEventListener("click", () => { void goTo(currentPage - 1); });
el.pageNext.addEventListener("click", () => { void goTo(currentPage + 1); });
el.pageInput.addEventListener("change", () => { void goTo(Number(el.pageInput.value)); });

function stepZoom(direction) {
    if (!document_) return;

    // '맞춤'에서 확대를 누르면 지금 보이는 크기에서 한 단계 올라가야 자연스럽다.
    // 그래서 마지막으로 실제 그린 배율을 기준으로 가장 가까운 단계를 찾는다.
    const current = zoom === "fit" ? lastScale : zoom;
    const nearest = ZOOM_STEPS.reduce((best, step) =>
        Math.abs(step - current) < Math.abs(best - current) ? step : best, ZOOM_STEPS[0]);

    // 지금 크기가 단계 사이에 있으면 그 단계로 가는 것만으로도 방향이 맞을 수 있다.
    let index = ZOOM_STEPS.indexOf(nearest);
    if (direction > 0 && nearest <= current) index += 1;
    else if (direction < 0 && nearest >= current) index -= 1;

    zoom = ZOOM_STEPS[Math.min(Math.max(index, 0), ZOOM_STEPS.length - 1)];
    void drawPage(currentPage);
}

el.zoomIn.addEventListener("click", () => stepZoom(1));
el.zoomOut.addEventListener("click", () => stepZoom(-1));
el.zoomFit.addEventListener("click", () => {
    if (!document_) return;
    zoom = "fit";
    void drawPage(currentPage);
});

// 창 크기가 바뀌면 '쪽 맞춤'일 때만 다시 그린다. 사용자가 직접 고른 배율은 지킨다.
let resizeTimer = 0;
window.addEventListener("resize", () => {
    if (!document_ || zoom !== "fit") return;
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => { void drawPage(currentPage); }, 150);
});

// app.js가 쓸 수 있게 밖으로 내어 둔다.
window.PdfViewer = {
    open,
    close,
    isOpen: () => document_ !== null,
    pageCount: () => pageCount,
};
window.dispatchEvent(new Event("pdfviewer-ready"));
