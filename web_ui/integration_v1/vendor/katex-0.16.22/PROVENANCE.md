# KaTeX 0.16.22 vendored runtime

This directory contains the browser runtime needed by `web_ui/integration_v1`.
It was fetched only from the official npm registry with:

```text
npm pack katex@0.16.22 --registry=https://registry.npmjs.org --ignore-scripts
```

- Package: `katex@0.16.22`
- Registry tarball: `https://registry.npmjs.org/katex/-/katex-0.16.22.tgz`
- npm `dist.integrity`: `sha512-XCHRdUw4lf3SKBaJe4EvgqIuWwkPSo9XoeO8GjQW94Bp7TWv9hNhzZjZ+OH9yf1UmLygb7DIT5GSFQiyt16zYg==`
- License: MIT; exact upstream text is in `LICENSE`.
- Included runtime files: `katex.min.js`, `katex.min.css`, and the CSS-referenced
  font assets under `fonts/`.

The extracted package metadata was checked for the exact package name and
version before these assets were copied. No CDN is used at runtime.
