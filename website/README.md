# Docs site

The sqlite-predict documentation site, built with [Astro](https://astro.build)
and [Starlight](https://starlight.astro.build). Content lives in
`src/content/docs/`; the sidebar and site metadata are in `astro.config.mjs`.

```sh
npm install
npm run dev      # local dev server at http://localhost:4321
npm run build    # static build into dist/
npm run preview  # serve the built site
```

## Deploying

`dist/` is a static site. Cloudflare Pages is the recommended host while the repo
is private (GitHub Pages needs a public repo or a paid plan). Set `site` in
`astro.config.mjs` to the deployed URL so canonical links and the sitemap are
correct.
