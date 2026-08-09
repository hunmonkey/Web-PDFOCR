import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { build, version as esbuildVersion } from "esbuild";

const EXPECTED_SDK_VERSION = "0.4.2";
const EXPECTED_ORT_VERSION = "1.27.0";
const EXPECTED_ESBUILD_VERSION = "0.28.2";
const PATCH_ID = "paddleocr-sequential-det-then-rec-v1";

const browserDir = path.dirname(fileURLToPath(import.meta.url));
const outputDir = path.join(browserDir, "dist");
const outputFile = path.join(outputDir, "paddleocr-browser.mjs");

const sdkEntry = fileURLToPath(import.meta.resolve("@paddleocr/paddleocr-js"));
const ortWebgpuEntry = fileURLToPath(import.meta.resolve("onnxruntime-web/webgpu"));

const sdkPackage = JSON.parse(
  await readFile(path.join(path.dirname(sdkEntry), "..", "package.json"), "utf8"),
);
const ortPackage = JSON.parse(
  await readFile(path.join(path.dirname(ortWebgpuEntry), "..", "package.json"), "utf8"),
);

if (sdkPackage.version !== EXPECTED_SDK_VERSION) {
  throw new Error(
    `PaddleOCR.js version mismatch: expected ${EXPECTED_SDK_VERSION}, got ${sdkPackage.version}`,
  );
}
if (ortPackage.version !== EXPECTED_ORT_VERSION) {
  throw new Error(
    `ONNX Runtime Web version mismatch: expected ${EXPECTED_ORT_VERSION}, got ${ortPackage.version}`,
  );
}
if (esbuildVersion !== EXPECTED_ESBUILD_VERSION) {
  throw new Error(
    `esbuild version mismatch: expected ${EXPECTED_ESBUILD_VERSION}, got ${esbuildVersion}`,
  );
}

const originalSessionCreation = `    const [detModel, recModel] = await Promise.all([
      createDetModel({
        ort: this.ort,
        modelBytes: loadedAssets[0].modelBytes,
        configText: loadedAssets[0].configText,
        backend,
        webgpuState,
        batchSize: detBatchSize
      }),
      createRecModel({
        ort: this.ort,
        modelBytes: loadedAssets[1].modelBytes,
        configText: loadedAssets[1].configText,
        backend,
        webgpuState,
        batchSize: recBatchSize
      })
    ]);`;

const sequentialSessionCreation = `    const detModel = await createDetModel({
      ort: this.ort,
      modelBytes: loadedAssets[0].modelBytes,
      configText: loadedAssets[0].configText,
      backend,
      webgpuState,
      batchSize: detBatchSize
    });
    let recModel;
    try {
      recModel = await createRecModel({
        ort: this.ort,
        modelBytes: loadedAssets[1].modelBytes,
        configText: loadedAssets[1].configText,
        backend,
        webgpuState,
        batchSize: recBatchSize
      });
    } catch (error) {
      await detModel.dispose().catch(() => {});
      throw error;
    }`;

const sdkSource = await readFile(sdkEntry, "utf8");
const originalMatchCount = sdkSource.split(originalSessionCreation).length - 1;
if (originalMatchCount !== 1) {
  throw new Error(
    `PaddleOCR.js session patch target changed: expected exactly 1 match, got ${originalMatchCount}`,
  );
}

const patchedSdkSource = sdkSource.replace(
  originalSessionCreation,
  sequentialSessionCreation,
);
if (patchedSdkSource.includes(originalSessionCreation)) {
  throw new Error("PaddleOCR.js concurrent session creation remained after patching");
}
const sequentialMatchCount = patchedSdkSource.split(sequentialSessionCreation).length - 1;
if (sequentialMatchCount !== 1) {
  throw new Error(
    `PaddleOCR.js sequential session patch was not exact: got ${sequentialMatchCount} matches`,
  );
}

const buildInfo = {
  paddleocr: EXPECTED_SDK_VERSION,
  onnxruntimeWeb: EXPECTED_ORT_VERSION,
  onnxruntimeEntry: "onnxruntime-web/webgpu",
  patch: PATCH_ID,
};
const entrySource = `export const PDFOCR_BROWSER_BUILD_INFO = Object.freeze(${JSON.stringify(buildInfo)});\n${patchedSdkSource}`;

await mkdir(outputDir, { recursive: true });
const result = await build({
  absWorkingDir: browserDir,
  bundle: true,
  external: ["crypto", "fs", "path", "node:crypto", "node:fs", "node:path"],
  format: "esm",
  legalComments: "eof",
  metafile: true,
  minify: true,
  outfile: outputFile,
  platform: "browser",
  sourcemap: false,
  splitting: false,
  stdin: {
    contents: entrySource,
    loader: "js",
    resolveDir: path.dirname(sdkEntry),
    sourcefile: "paddleocr-patched.mjs",
  },
  target: ["es2022"],
  treeShaking: true,
  write: false,
  alias: {
    "onnxruntime-web": "onnxruntime-web/webgpu",
  },
});

const inputNames = Object.keys(result.metafile.inputs).map((name) =>
  name.replaceAll("\\", "/"),
);
if (
  !inputNames.some((name) =>
    name.endsWith("node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs"),
  )
) {
  throw new Error("The bundle did not resolve ONNX Runtime Web through its WebGPU entry");
}
if (Object.keys(result.metafile.outputs).length !== 1) {
  throw new Error("The browser build must emit exactly one ESM file");
}
if (result.outputFiles.length !== 1) {
  throw new Error("The browser build produced an unexpected extra output file");
}

const outputMetadata = Object.values(result.metafile.outputs)[0];
const allowedExternalImports = new Set([
  "crypto",
  "fs",
  "path",
  "node:crypto",
  "node:fs",
  "node:path",
]);
const unexpectedImports = outputMetadata.imports.filter(
  (item) =>
    !(
      item.external &&
      item.kind === "require-call" &&
      allowedExternalImports.has(item.path)
    ),
);
if (unexpectedImports.length !== 0) {
  throw new Error(
    `The single-file bundle retained unexpected imports: ${JSON.stringify(unexpectedImports)}`,
  );
}

const bundleSource = result.outputFiles[0].text;
for (const marker of [
  "ONNX Runtime Web v1.27.0",
  "PDFOCR_BROWSER_BUILD_INFO",
  "PaddleOCR",
  PATCH_ID,
  "onnxruntime-web/webgpu",
]) {
  if (!bundleSource.includes(marker)) {
    throw new Error(`Browser bundle verification marker is missing: ${marker}`);
  }
}

await writeFile(outputFile, result.outputFiles[0].contents);

console.log(
  `Built ${path.relative(browserDir, outputFile)} (${Buffer.byteLength(bundleSource).toLocaleString("en-US")} bytes)`,
);
console.log(
  `Verified PaddleOCR.js ${EXPECTED_SDK_VERSION}, ONNX Runtime Web ${EXPECTED_ORT_VERSION} WebGPU, and ${PATCH_ID}`,
);
