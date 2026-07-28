// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// site: set to the deployed URL for correct canonical links + sitemap.
// (Cloudflare Pages recommended for a private repo; GitHub Pages needs a public
// repo or a paid plan.)
export default defineConfig({
  site: "https://sqlite-predict.pages.dev",
  integrations: [
    starlight({
      title: "sqlite-predict",
      description:
        "Forecasting, anomaly detection, and prediction as SQL primitives for SQLite, with replayable receipts.",
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
            {
              label: "Auto-selection & conformal intervals",
              link: "/guides/auto-and-conformal/",
            },
            { label: "Backtesting", link: "/guides/backtesting/" },
            { label: "Receipts & replay", link: "/guides/receipts/" },
            { label: "Distillation", link: "/guides/distillation/" },
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
