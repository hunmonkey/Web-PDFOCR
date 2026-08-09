// =========================================================================
// 브라우저 그래픽카드 OCR
//
// 서버가 한 쪽씩 그려 준 PNG를 공식 PaddleOCR.js로 읽고, 글자와 위치만
// 서버에 돌려보낸다. 최종 PDF 조립과 저장은 기존 서버가 맡는다.
// =========================================================================

const BUNDLE_URL = "/static/vendor/paddleocr-browser.mjs";
const DET_MODEL_URL = "/static/vendor/PP-OCRv6_small_det_onnx_infer.tar";
const REC_MODEL_URL = "/static/vendor/korean_PP-OCRv5_mobile_rec_onnx_infer.tar";
const WASM_PATH = "/static/vendor/";
const MAX_IMAGE_SIDE = 2000;

let ocrPromise = null;

function abortIfNeeded(signal) {
    if (signal && signal.aborted) {
        throw new DOMException("사용자가 작업을 중지했습니다.", "AbortError");
    }
}

async function responseError(response, fallback) {
    try {
        const data = await response.json();
        return new Error(data.detail || fallback);
    } catch (_) {
        return new Error(fallback);
    }
}

function retryableStatus(status) {
    return status === 408 || status === 425 || status === 429 || status >= 500;
}

function retryDelay(milliseconds, signal) {
    return new Promise((resolve, reject) => {
        const finish = () => {
            if (signal) signal.removeEventListener("abort", onAbort);
            resolve();
        };
        const timer = setTimeout(finish, milliseconds);
        const onAbort = () => {
            clearTimeout(timer);
            reject(new DOMException("사용자가 작업을 중지했습니다.", "AbortError"));
        };
        if (signal) signal.addEventListener("abort", onAbort, { once: true });
    });
}

async function fetchWithRetry(url, options, attempts = 3) {
    let lastError = null;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
        abortIfNeeded(options.signal);
        try {
            const response = await fetch(url, options);
            if (!retryableStatus(response.status) || attempt === attempts - 1) {
                return response;
            }
            if (response.body) await response.body.cancel();
        } catch (error) {
            if (error && error.name === "AbortError") throw error;
            lastError = error;
            if (attempt === attempts - 1) throw error;
        }
        await retryDelay(250 * (attempt + 1), options.signal);
    }
    throw lastError || new Error("서버에 연결하지 못했습니다.");
}

async function finishBrowserJob(jobId, signal, onStage) {
    while (true) {
        abortIfNeeded(signal);
        let response;
        try {
            response = await fetchWithRetry("/api/jobs/" + jobId + "/finish", {
                method: "POST",
                signal,
            });
        } catch (error) {
            if (error && error.name === "AbortError") throw error;
            onStage("서버 연결이 잠시 끊겼습니다. 결과 PDF 마무리를 다시 요청하는 중입니다.");
            await retryDelay(1000, signal);
            continue;
        }

        if (response.ok) return;
        if (!retryableStatus(response.status)) {
            throw await responseError(response, "결과 PDF 마무리를 시작하지 못했습니다.");
        }

        // 서버가 요청을 이미 받았지만 중간 장비가 5xx 응답을 만들었을 수도 있다.
        // finish는 서버에서 같은 요청을 여러 번 받아도 안전하므로 다시 확인한다.
        if (response.body) await response.body.cancel();
        onStage("서버가 결과 PDF 마무리 요청을 확인하는 중입니다.");
        await retryDelay(1000, signal);
    }
}

function usesWebGpu(provider) {
    return String(provider || "").toLowerCase().includes("webgpu");
}

async function createOcr() {
    if (!window.isSecureContext || !navigator.gpu) {
        throw new Error(
            "이 브라우저에서는 그래픽카드 인식을 사용할 수 없습니다. " +
            "최신 Chrome 또는 Edge에서 HTTPS 주소로 접속해 주세요."
        );
    }

    const { PaddleOCR } = await import(BUNDLE_URL);
    const ocr = await PaddleOCR.create({
        worker: false,
        textDetectionModelName: "PP-OCRv6_small_det",
        textDetectionModelAsset: { url: DET_MODEL_URL },
        textRecognitionModelName: "korean_PP-OCRv5_mobile_rec",
        textRecognitionModelAsset: { url: REC_MODEL_URL },
        textDetectionBatchSize: 1,
        textRecognitionBatchSize: 6,
        textDetLimitSideLen: MAX_IMAGE_SIDE,
        textDetLimitType: "max",
        textDetMaxSideLimit: MAX_IMAGE_SIDE,
        textDetThresh: 0.3,
        textDetBoxThresh: 0.5,
        textDetUnclipRatio: 1.6,
        textRecScoreThresh: 0,
        unsupportedBehavior: "ignore",
        ortOptions: {
            backend: "webgpu",
            wasmPaths: WASM_PATH,
            // 별도 교차 출처 격리 설정 없이도 쓸 수 있게 한 갈래로 실행한다.
            numThreads: 1,
            simd: true,
        },
    });

    const summary = ocr.getInitializationSummary();
    if (!summary || !usesWebGpu(summary.detProvider) || !usesWebGpu(summary.recProvider)) {
        await ocr.dispose();
        throw new Error(
            "그래픽카드 인식 도구를 준비하지 못했습니다. 서버에서 읽기를 선택해 주세요."
        );
    }
    return ocr;
}

async function getOcr() {
    if (!ocrPromise) {
        ocrPromise = createOcr().catch((error) => {
            // 일시적인 내려받기 실패 뒤 다시 시도할 수 있어야 한다.
            ocrPromise = null;
            throw error;
        });
    }
    return ocrPromise;
}

async function discardOcr(ocr) {
    ocrPromise = null;
    try {
        await ocr.dispose();
    } catch (_) {
        // 이미 그래픽 장치 연결이 끊긴 경우 dispose도 실패할 수 있다.
    }
}

function numberHeader(response, name) {
    const value = Number.parseInt(response.headers.get(name) || "", 10);
    if (!Number.isInteger(value)) {
        throw new Error("서버가 보낸 페이지 정보가 올바르지 않습니다.");
    }
    return value;
}

function clamp(value, maximum) {
    return Math.min(maximum, Math.max(0, Number(value)));
}

function normalizedLine(item, width, height, textScore) {
    if (!item || typeof item.text !== "string" || !item.text.trim()) return null;
    if (!Number.isFinite(item.score) || item.score < textScore) return null;
    if (!Array.isArray(item.poly) || item.poly.length !== 4) return null;

    const box = [];
    for (const point of item.poly) {
        if (!Array.isArray(point) || point.length !== 2) return null;
        if (!Number.isFinite(point[0]) || !Number.isFinite(point[1])) return null;
        box.push([clamp(point[0], width), clamp(point[1], height)]);
    }

    const xs = box.map((point) => point[0]);
    const ys = box.map((point) => point[1]);
    if (Math.max(...xs) <= Math.min(...xs) || Math.max(...ys) <= Math.min(...ys)) {
        return null;
    }
    return { box, text: item.text, score: item.score };
}

async function readPage(ocr, jobId, pageIndex, textScore, signal) {
    abortIfNeeded(signal);
    const imageResponse = await fetchWithRetry(
        "/api/jobs/" + jobId + "/pages/" + pageIndex + "/image",
        { signal, cache: "no-store" }
    );
    if (!imageResponse.ok) {
        throw await responseError(imageResponse, "PDF 쪽 이미지를 받지 못했습니다.");
    }

    const width = numberHeader(imageResponse, "X-PDFOCR-Width");
    const height = numberHeader(imageResponse, "X-PDFOCR-Height");
    const rotation = numberHeader(imageResponse, "X-PDFOCR-Rotation");
    const image = await imageResponse.blob();

    abortIfNeeded(signal);
    let results;
    try {
        results = await ocr.predict(image, {
            textDetLimitSideLen: MAX_IMAGE_SIDE,
            textDetLimitType: "max",
            textDetMaxSideLimit: MAX_IMAGE_SIDE,
            textRecScoreThresh: 0,
        });
    } catch (error) {
        await discardOcr(ocr);
        throw error;
    }
    abortIfNeeded(signal);

    const result = results[0];
    if (!result || result.image.width !== width || result.image.height !== height) {
        await discardOcr(ocr);
        throw new Error("글자를 읽은 페이지 크기가 원본과 다릅니다.");
    }
    if (!usesWebGpu(result.runtime.detProvider) || !usesWebGpu(result.runtime.recProvider)) {
        await discardOcr(ocr);
        throw new Error("그래픽카드 대신 다른 장치가 선택되어 작업을 중지했습니다.");
    }

    const lines = result.items
        .map((item) => normalizedLine(item, width, height, textScore))
        .filter((item) => item !== null);

    const saveResponse = await fetchWithRetry(
        "/api/jobs/" + jobId + "/pages/" + pageIndex,
        {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                render_width: width,
                render_height: height,
                rotation,
                lines,
            }),
            signal,
        }
    );
    if (!saveResponse.ok) {
        throw await responseError(saveResponse, "인식한 글자를 서버에 전달하지 못했습니다.");
    }
    return result.metrics;
}

export async function runBrowserOcr({
    jobId,
    totalPages,
    textScore,
    signal,
    onStage = () => {},
}) {
    abortIfNeeded(signal);
    onStage("그래픽카드 인식 도구를 준비하는 중입니다. 처음에는 약 60MB를 받습니다.");
    const ocr = await getOcr();
    abortIfNeeded(signal);

    const metrics = [];
    for (let pageIndex = 0; pageIndex < totalPages; pageIndex += 1) {
        onStage(
            "전체 " + totalPages + "쪽 중 " + (pageIndex + 1) + "쪽을 이 컴퓨터에서 읽는 중입니다."
        );
        metrics.push(await readPage(ocr, jobId, pageIndex, textScore, signal));
    }

    abortIfNeeded(signal);
    onStage("인식한 글자를 PDF에 넣도록 서버에 요청하는 중입니다.");
    await finishBrowserJob(jobId, signal, onStage);

    return metrics;
}
