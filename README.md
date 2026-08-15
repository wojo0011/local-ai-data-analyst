# Local AI Data Analyst

A portfolio-grade, browser-only data analysis lab that turns a CSV into a local analytics workflow:

```text
CSV
 │
 ▼
DuckDB-Wasm
 │
 ├── SQL + deterministic statistics
 │
 ▼
Transformers.js (semantic schema search)
 │
 ▼
WebLLM (optional local analyst)
 │
 ▼
D3 / Apache ECharts
```

## Why this architecture

The project intentionally keeps **deterministic computation separate from language-model interpretation**. DuckDB-Wasm is the numeric source of truth. Statistics explain the evidence mathematically. Transformers.js helps map natural-language questions to relevant columns. WebLLM receives a compact local evidence context and explains findings without becoming the calculator.

## Features

- CSV upload, drag/drop, pasted CSV, and a clearly labelled synthetic retail demo dataset
- DuckDB-Wasm schema inference, profiling, missingness, cardinality and read-only SQL lab
- Descriptive statistics, quartiles/IQR, outlier diagnostics, Pearson correlation and OLS regression
- KaTeX equations using the same math-library family used by BrowserML Studio
- ECharts scatter, line, grouped-bar and histogram views
- D3 architecture visualization
- Optional Transformers.js semantic schema search using local embeddings
- Optional WebLLM analyst running in-browser with WebGPU
- Shared `vis_mode` light/dark theme behavior, portfolio neural loader and universal portfolio footer
- No application backend; compatible with GitHub Pages

## Privacy model

The application does not implement a data-analysis API or upload endpoint. CSV parsing, SQL and statistics run inside the browser. Optional model weights are downloaded from public model/package hosts. The app sends a compact dataset context only to a WebLLM model running in the same browser process.

The SQL lab is intentionally restricted to read-only analytical statements to prevent export/mutation commands from undermining the local-data privacy model.

## Mathematical scope

The interface explains and calculates:

- arithmetic mean: `x̄ = (1/n) Σxᵢ`
- sample standard deviation: `s = √(Σ(xᵢ-x̄)²/(n-1))`
- median and quartiles
- IQR outlier fences: `[Q1 - 1.5 IQR, Q3 + 1.5 IQR]`
- Pearson correlation: `r = cov(X,Y)/(sX sY)`
- ordinary least squares: `ŷ = β₀ + β₁x`, plus `R²`
- cosine similarity for semantic schema ranking

These are descriptive/associational tools. The app explicitly avoids treating correlation or regression as causal evidence.

## Run locally

Because ES modules, WebAssembly workers and model downloads need an HTTP origin, use any static server instead of opening `index.html` directly:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Tests

No test dependencies are required:

```bash
npm test
npm run check
```

The unit tests validate descriptive statistics, sample variance/standard deviation, Pearson correlation, OLS regression, IQR outlier bounds, histogram conservation and cosine similarity.

## Browser/runtime notes

- Core CSV + DuckDB + statistics do not require WebGPU.
- Transformers.js can use WebGPU when available and can fall back to its browser runtime.
- WebLLM requires WebGPU and can require hundreds of megabytes of model/runtime data.
- DuckDB-Wasm/browser memory limits still apply to large datasets.

## Primary technical references

- DuckDB-Wasm instantiation and data ingestion: DuckDB documentation
- Transformers.js pipeline API and WebGPU guide: Hugging Face documentation
- WebLLM basic usage and model loading: MLC WebLLM documentation
- Apache ECharts browser/CDN usage: Apache ECharts handbook

## Portfolio design system

The UI follows the shared `wojtekmatwiejczyk.ca` design tokens: Teko + Manrope, `#252525` dark paper, `#ffffff` light paper, `#eef2f7` dark-mode ink, `#1f2933` light-mode ink, and the portfolio gold accents `#e7b532` / `#e9a70c`. The site also reuses the shared portfolio loader and universal footer assets.
