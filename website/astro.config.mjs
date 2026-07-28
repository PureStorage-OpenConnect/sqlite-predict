// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// GitHub Pages serves this project at different URLs depending on visibility: a
// random *.pages.github.io ROOT while the repo is internal (private Pages), and
// purestorage-openconnect.github.io/sqlite-predict once it is public. The deploy
// workflow reads the live Pages config (actions/configure-pages) and passes the
// correct origin + base in via env, so one build works for either URL. Locally
// we default to root. Internal doc links are path-relative, so only asset and
// nav paths depend on `base`.
const base = process.env.PAGES_BASE_PATH || "/";
const site = process.env.PAGES_ORIGIN || "https://purestorage-openconnect.github.io";

export default defineConfig({
  site,
  base,
  integrations: [
    starlight({
      title: "sqlite-predict",
      description:
        "Forecasting, anomaly detection, and prediction as SQL primitives for SQLite.",
      customCss: ["./src/styles/theme.css"],
      // Pages are short; the "On this page" TOC adds clutter and forces a third
      // column. Drop it site-wide for a clean nav + content layout.
      tableOfContents: false,
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/PureStorage-OpenConnect/sqlite-predict",
        },
      ],
      editLink: {
        baseUrl:
          "https://github.com/PureStorage-OpenConnect/sqlite-predict/edit/main/website/",
      },
      sidebar: [
        {
          label: "Start here",
          items: [
            { label: "Introduction", link: "/" },
            { label: "Python", link: "/getting-started/python/" },
            { label: "JavaScript / Node", link: "/getting-started/javascript/" },
            { label: "Rust", link: "/getting-started/rust/" },
            { label: "CLI / C", link: "/getting-started/cli/" },
          ],
        },
        {
          label: "Guides",
          items: [
            { label: "Operations", link: "/guides/operations/" },
            { label: "Distillation", link: "/guides/distillation/" },
            {
              label: "Auto-selection & conformal intervals",
              link: "/guides/auto-and-conformal/",
            },
            { label: "Backtesting", link: "/guides/backtesting/" },
            { label: "Using with ORMs", link: "/guides/orms/" },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "Functions", link: "/reference/functions/" },
            { label: "Models", link: "/reference/models/" },
            { label: "Options", link: "/reference/options/" },
            { label: "Errors", link: "/reference/errors/" },
          ],
        },
        { label: "Benchmarks", link: "/benchmarks/" },
      ],
    }),
  ],
});
