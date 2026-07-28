/* SPDX-License-Identifier: MIT OR Apache-2.0
 * Copyright (c) 2026 Pure Storage, Inc.
 */
/* forecast(query, horizon, options) — eponymous table-valued function.
 * RFC 0005 §4.2.1. Vtab shape follows SQLite's ext/misc/series.c. */
#include "predict-internal.h"
#include "predict-student.h"

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT3
#endif

#define FORECAST_MAX_HORIZON 1000
#define FORECAST_DEFAULT_CONTEXT_LIMIT 4096
#define FORECAST_MIN_HISTORY 8
#define FORECAST_MAX_TOTAL_ROWS (1 << 21)
#define FORECAST_MAX_GROUP_COLS 8

/* column order must match the CREATE TABLE below */
#define FC_COL_SERIES_KEY 0
#define FC_COL_STEP 1
#define FC_COL_TS 2
#define FC_COL_FORECAST 3
#define FC_COL_LOWER 4
#define FC_COL_UPPER 5
#define FC_COL_STATUS 6
#define FC_COL_RECEIPT 7
#define FC_COL_QUERY 8
#define FC_COL_HORIZON 9
#define FC_COL_OPTIONS 10

#pragma region models

/* Detect a seasonal period: autocorrelation of the OLS-detrended series
 * (a trend inflates raw ACF at every lag and buries the seasonal peak),
 * accepting only local ACF peaks above threshold. Returns 0 when nothing
 * convincing is found. */
static int detect_period(const f64 *y, int n) {
  if (n < 3 * 2)
    return 0;
  int max_lag = n / 2;
  if (max_lag > 366)
    max_lag = 366;

  f64 *d = sqlite3_malloc(sizeof(f64) * n);
  if (!d)
    return 0;
  f64 st = 0, sy = 0, stt = 0, sty = 0;
  for (int i = 0; i < n; i++) {
    st += i;
    sy += y[i];
    stt += (f64)i * i;
    sty += (f64)i * y[i];
  }
  f64 denom = (f64)n * stt - st * st;
  f64 b = denom != 0 ? ((f64)n * sty - st * sy) / denom : 0;
  f64 a = (sy - b * st) / n;
  for (int i = 0; i < n; i++)
    d[i] = y[i] - (a + b * i);

  f64 c0 = 0;
  for (int i = 0; i < n; i++)
    c0 += d[i] * d[i];
  if (c0 <= 0) {
    sqlite3_free(d);
    return 0;
  }

  f64 *acf = sqlite3_malloc(sizeof(f64) * (max_lag + 2));
  if (!acf) {
    sqlite3_free(d);
    return 0;
  }
  for (int lag = 1; lag <= max_lag + 1 && lag < n; lag++) {
    f64 c = 0;
    for (int i = lag; i < n; i++)
      c += d[i] * d[i - lag];
    acf[lag] = c / c0;
  }

  int best = 0;
  f64 best_acf = 0.3; /* threshold: below this, treat as non-seasonal */
  for (int lag = 2; lag <= max_lag && lag + 1 < n; lag++) {
    int is_peak = acf[lag] > acf[lag - 1] && acf[lag] >= acf[lag + 1];
    if (is_peak && acf[lag] > best_acf) {
      best_acf = acf[lag];
      best = lag;
    }
  }
  sqlite3_free(d);
  sqlite3_free(acf);
  return best;
}

/* Seasonal naive with drift. Baseline floor: repeat last season (or last
 * value) plus the global drift. */
static void model_seasonal_naive(const f64 *y, int n, int horizon, f64 *fc,
                                 f64 *sigma) {
  int p = detect_period(y, n);
  f64 drift = n > 1 ? (y[n - 1] - y[0]) / (n - 1) : 0;

  for (int h = 1; h <= horizon; h++) {
    f64 base;
    i64 gap; /* steps between the base observation and the target */
    if (p > 0 && n >= p) {
      int base_idx = n - p + ((h - 1) % p);
      base = y[base_idx];
      gap = (n - 1 + h) - base_idx;
    } else {
      base = y[n - 1];
      gap = h;
    }
    fc[h - 1] = base + drift * (f64)gap;
  }

  if (!sigma)
    return; /* point-only path (backtesting): skip the interval computation */

  /* Per-horizon sigma from an in-sample backtest of this exact forecast:
   * for each past origin, forecast h steps ahead and record the error, then
   * take the RMS per h. This calibrates the interval to the model's real
   * h-step error, with no lag-1-vs-seasonal scale confusion and no random-walk
   * sqrt(h) growth assumption (both of which made the old intervals too wide
   * on seasonal data). Falls back to lag-1 * sqrt(h) only on OOM. */
  f64 *ss = sqlite3_malloc(sizeof(f64) * horizon);
  int *cnt = sqlite3_malloc(sizeof(int) * horizon);
  if (ss && cnt) {
    for (int h = 0; h < horizon; h++) {
      ss[h] = 0;
      cnt[h] = 0;
    }
    int t0 = p > 0 ? p : 1;
    for (int t = t0; t < n; t++) { /* history y[0..t-1], forecast y[t..] */
      f64 dr = t > 1 ? (y[t - 1] - y[0]) / (t - 1) : 0;
      for (int h = 1; h <= horizon && t - 1 + h < n; h++) {
        f64 base;
        i64 gap;
        if (p > 0 && t >= p) {
          int bi = t - p + ((h - 1) % p);
          base = y[bi];
          gap = (t - 1 + h) - bi;
        } else {
          base = y[t - 1];
          gap = h;
        }
        f64 e = y[t - 1 + h] - (base + dr * (f64)gap);
        ss[h - 1] += e * e;
        cnt[h - 1]++;
      }
    }
    f64 fb = 0; /* lag-1 fallback scale for thin horizons */
    int fm = 0;
    for (int i = 1; i < n; i++) {
      f64 r = y[i] - y[i - 1];
      fb += r * r;
      fm++;
    }
    fb = fm > 0 ? sqrt(fb / fm) : 0;
    for (int h = 1; h <= horizon; h++)
      sigma[h - 1] =
          cnt[h - 1] >= 2 ? sqrt(ss[h - 1] / cnt[h - 1]) : fb * sqrt((f64)h);
  } else {
    f64 s2 = 0;
    int m = 0;
    for (int i = 1; i < n; i++) {
      f64 r = y[i] - y[i - 1];
      s2 += r * r;
      m++;
    }
    f64 sigma1 = m > 0 ? sqrt(s2 / m) : 0;
    for (int h = 1; h <= horizon; h++)
      sigma[h - 1] = sigma1 * sqrt((f64)h);
  }
  sqlite3_free(ss);
  sqlite3_free(cnt);
}

/* Classic Theta(0,2) with additive seasonal adjustment: average of the
 * linear-regression line extrapolation and SES (grid-searched alpha) on
 * the theta=2 line. The strongest of the simple statistical methods. */
static void model_theta(const f64 *y, int n, int horizon, f64 *fc,
                        f64 *sigma) {
  int p = detect_period(y, n);
  f64 *yd = sqlite3_malloc(sizeof(f64) * n);
  f64 *seas = NULL;
  if (!yd) {
    model_seasonal_naive(y, n, horizon, fc, sigma);
    return;
  }
  memcpy(yd, y, sizeof(f64) * n);

  /* additive seasonal indices from period-phase means of the DETRENDED
   * series — computing them on raw values leaks the trend slope into
   * the indices, one phase-width of drift per index */
  if (p > 1 && n >= 2 * p) {
    seas = sqlite3_malloc(sizeof(f64) * p);
    if (seas) {
      f64 st0 = 0, sy0 = 0, stt0 = 0, sty0 = 0;
      for (int i = 0; i < n; i++) {
        st0 += i;
        sy0 += y[i];
        stt0 += (f64)i * i;
        sty0 += (f64)i * y[i];
      }
      f64 den0 = (f64)n * stt0 - st0 * st0;
      f64 b0 = den0 != 0 ? ((f64)n * sty0 - st0 * sy0) / den0 : 0;
      f64 a0 = (sy0 - b0 * st0) / n;
      for (int k = 0; k < p; k++) {
        f64 s = 0;
        int c = 0;
        for (int i = k; i < n; i += p) {
          s += y[i] - (a0 + b0 * i);
          c++;
        }
        seas[k] = c ? s / c : 0;
      }
      for (int i = 0; i < n; i++)
        yd[i] -= seas[i % p];
    }
  }

  /* OLS line a + b*t over the deseasonalized series */
  f64 st = 0, sy = 0, stt = 0, sty = 0;
  for (int i = 0; i < n; i++) {
    st += i;
    sy += yd[i];
    stt += (f64)i * i;
    sty += i * yd[i];
  }
  f64 denom = n * stt - st * st;
  f64 b = denom != 0 ? (n * sty - st * sy) / denom : 0;
  f64 a = (sy - b * st) / n;

  /* SES over the theta=2 line, alpha by grid search on one-step SSE */
  f64 best_alpha = 0.5, best_sse = -1, best_level = yd[n - 1];
  for (f64 alpha = 0.05; alpha < 0.96; alpha += 0.05) {
    f64 level = 2 * yd[0] - (a);
    f64 sse = 0;
    for (int i = 1; i < n; i++) {
      f64 z = 2 * yd[i] - (a + b * i);
      f64 e = z - level;
      sse += e * e;
      level += alpha * e;
    }
    if (best_sse < 0 || sse < best_sse) {
      best_sse = sse;
      best_alpha = alpha;
      best_level = level;
    }
  }
  f64 sigma1 = n > 1 ? sqrt(best_sse / (n - 1)) / 2 : 0;

  for (int h = 1; h <= horizon; h++) {
    f64 line = a + b * (n - 1 + h);
    f64 theta_fc = 0.5 * line + 0.5 * best_level;
    if (seas && p > 0)
      theta_fc += seas[(n + h - 1) % p];
    fc[h - 1] = theta_fc;
  }

  if (!sigma) { /* point-only path (backtesting): skip the interval computation */
    sqlite3_free(yd);
    sqlite3_free(seas);
    return;
  }

  /* Per-horizon sigma from the fitted model's in-sample h-step errors: roll
   * the theta forecast forward from every past origin using that origin's SES
   * level, and take the RMS error per h. Empirically calibrated per horizon,
   * replacing the old ad-hoc sigma1 * sqrt(h) / 2 growth. */
  f64 *lev = sqlite3_malloc(sizeof(f64) * n);
  f64 *ss = sqlite3_malloc(sizeof(f64) * horizon);
  int *cnt = sqlite3_malloc(sizeof(int) * horizon);
  if (lev && ss && cnt) {
    f64 level = 2 * yd[0] - a;
    lev[0] = level;
    for (int i = 1; i < n; i++) {
      f64 z = 2 * yd[i] - (a + b * i);
      level += best_alpha * (z - level);
      lev[i] = level;
    }
    for (int h = 0; h < horizon; h++) {
      ss[h] = 0;
      cnt[h] = 0;
    }
    for (int t = 0; t < n; t++)
      for (int h = 1; h <= horizon && t + h < n; h++) {
        f64 f = 0.5 * (a + b * (t + h)) + 0.5 * lev[t];
        if (seas && p > 0)
          f += seas[(t + h) % p];
        f64 e = y[t + h] - f;
        ss[h - 1] += e * e;
        cnt[h - 1]++;
      }
    for (int h = 1; h <= horizon; h++)
      sigma[h - 1] =
          cnt[h - 1] >= 2 ? sqrt(ss[h - 1] / cnt[h - 1]) : sigma1 * sqrt((f64)h);
  } else {
    for (int h = 1; h <= horizon; h++)
      sigma[h - 1] = sigma1 * sqrt((f64)h);
  }
  sqlite3_free(lev);
  sqlite3_free(ss);
  sqlite3_free(cnt);
  sqlite3_free(yd);
  sqlite3_free(seas);
}

/* Teunter-Syntetos-Babai (TSB) for intermittent demand: separate exponential
 * smoothing of the demand probability and the demand size, forecast = p*z (a
 * flat rate). Handles the sparse, mostly-zero series (rare events: errors,
 * retries, uncommon tool calls) that theta and seasonal-naive model poorly.
 * Non-negative demand is assumed; a value <= 0 counts as no demand. */
static void model_tsb(const f64 *y, int n, int horizon, f64 *fc, f64 *sigma) {
  const f64 alpha = 0.1, beta = 0.1; /* size and probability smoothing */
  f64 sz = 0, sq = 0;
  int nnz = 0;
  for (int i = 0; i < n; i++) {
    sq += y[i] * y[i];
    if (y[i] > 0) {
      sz += y[i];
      nnz++;
    }
  }
  f64 z = nnz ? sz / nnz : 0;   /* mean nonzero demand size */
  f64 p = n ? (f64)nnz / n : 0; /* demand probability */
  f64 *rate = sqlite3_malloc(sizeof(f64) * (n > 0 ? n : 1));
  f64 cur = p * z;
  for (int t = 0; t < n; t++) {
    if (y[t] > 0) {
      z += alpha * (y[t] - z);
      p += beta * (1.0 - p);
    } else {
      p += beta * (0.0 - p);
    }
    cur = p * z;
    if (rate)
      rate[t] = cur; /* causal forecast made after seeing y[0..t] */
  }
  for (int h = 0; h < horizon; h++)
    fc[h] = cur; /* intermittent forecast is a flat rate */

  if (!sigma) { /* point-only path (backtesting): skip the interval computation */
    sqlite3_free(rate);
    return;
  }

  f64 *ss = sqlite3_malloc(sizeof(f64) * horizon);
  int *cnt = sqlite3_malloc(sizeof(int) * horizon);
  if (rate && ss && cnt) {
    for (int h = 0; h < horizon; h++) {
      ss[h] = 0;
      cnt[h] = 0;
    }
    f64 fb2 = 0;
    int fbm = 0;
    for (int t = 1; t < n; t++) { /* h-step errors from each causal origin */
      f64 r = rate[t - 1];
      f64 e0 = y[t] - r;
      fb2 += e0 * e0;
      fbm++;
      for (int h = 1; h <= horizon && t - 1 + h < n; h++) {
        f64 e = y[t - 1 + h] - r;
        ss[h - 1] += e * e;
        cnt[h - 1]++;
      }
    }
    f64 fb = fbm > 0 ? sqrt(fb2 / fbm) : 0;
    for (int h = 1; h <= horizon; h++)
      sigma[h - 1] = cnt[h - 1] >= 2 ? sqrt(ss[h - 1] / cnt[h - 1]) : fb;
  } else { /* OOM fallback: flat sigma from the overall series std */
    f64 smean = 0;
    for (int i = 0; i < n; i++)
      smean += y[i];
    f64 mean = n ? smean / n : 0;
    f64 var = n ? sq / n - mean * mean : 0;
    for (int h = 0; h < horizon; h++)
      sigma[h] = var > 0 ? sqrt(var) : 0;
  }
  sqlite3_free(rate);
  sqlite3_free(ss);
  sqlite3_free(cnt);
}

typedef void (*predict0_ts_model)(const f64 *, int, int, f64 *, f64 *);

typedef struct {
  const char *id;
  predict0_ts_model run;
} ts_model_entry;

static const ts_model_entry TS_MODELS[] = {
    {"stub-seasonal-naive", model_seasonal_naive},
    {"theta-classic", model_theta},
    {"tsb", model_tsb},
};

static predict0_ts_model resolve_ts_model(const char *name) {
  if (!name || strcmp(name, "default-ts") == 0)
    return model_theta;
  for (usize i = 0; i < countof(TS_MODELS); i++) {
    if (strcmp(TS_MODELS[i].id, name) == 0)
      return TS_MODELS[i].run;
  }
  return NULL;
}

static const char *resolve_ts_model_id(const char *name) {
  if (!name || strcmp(name, "default-ts") == 0)
    return "theta-classic";
  for (usize i = 0; i < countof(TS_MODELS); i++) {
    if (strcmp(TS_MODELS[i].id, name) == 0)
      return TS_MODELS[i].id;
  }
  return NULL;
}

/* ---- shared rolling-origin backtest (auto-selection, conformal, backtest()) ---- */

#define FORECAST_DEFAULT_FOLDS 20
#define FORECAST_MAX_FOLDS 512
#define CONFORMAL_MIN_FOLDS 3
#define FORECAST_MAX_CANDIDATES 8

/* Mean absolute one-step difference of y[0..n): the in-sample naive-forecast
 * MAE that scales MASE. 0 for a constant (or length < 2) series. */
static f64 naive_scale(const f64 *y, int n) {
  f64 s = 0;
  int m = 0;
  for (int i = 1; i < n; i++) {
    s += fabs(y[i] - y[i - 1]);
    m++;
  }
  return m > 0 ? s / m : 0;
}

static f64 metric_mae(const f64 *a, const f64 *f, int m) {
  f64 s = 0;
  for (int i = 0; i < m; i++)
    s += fabs(a[i] - f[i]);
  return m > 0 ? s / m : 0;
}

static f64 metric_rmse(const f64 *a, const f64 *f, int m) {
  f64 s = 0;
  for (int i = 0; i < m; i++) {
    f64 e = a[i] - f[i];
    s += e * e;
  }
  return m > 0 ? sqrt(s / m) : 0;
}

/* Symmetric MAPE in [0,2]. A pair with |a|+|f| == 0 is a perfect 0-vs-0 hit:
 * it contributes 0 and is still counted, so two identical zero series score 0. */
static f64 metric_smape(const f64 *a, const f64 *f, int m) {
  f64 s = 0;
  for (int i = 0; i < m; i++) {
    f64 d = fabs(a[i]) + fabs(f[i]);
    if (d > 0)
      s += 2.0 * fabs(a[i] - f[i]) / d;
  }
  return m > 0 ? s / m : 0;
}

/* Rolling-origin backtest of a point model over series y[0..n) with timestamps
 * ts. For each of up to `nfolds` contiguous origins ending at the latest usable
 * cutoff, fit `model` on the prefix and forecast gap+horizon steps, keeping the
 * last `horizon` (so `gap` points sit between train end and the first scored
 * target: a leakage guard). Fills act/fcst[folds*horizon] row-major by fold;
 * when sig != NULL, the model's per-horizon sigma too; when cutoff_ms != NULL,
 * each origin's last-training timestamp. Returns the number of folds evaluated
 * (0 if history is too short), or a negative SQLITE_ code on OOM. */
static int ts_backtest(predict0_ts_model model, const f64 *y, const i64 *ts,
                       int n, int horizon, int gap, int nfolds, f64 *act,
                       f64 *fcst, f64 *sig, i64 *cutoff_ms) {
  int H2 = gap + horizon;
  int c_max = n - H2;               /* latest cutoff with H2 held-out actuals */
  int c_min = FORECAST_MIN_HISTORY; /* minimum training length */
  if (c_max < c_min)
    return 0;
  int avail = c_max - c_min + 1;
  int folds = nfolds < avail ? nfolds : avail;
  f64 *fb = sqlite3_malloc(sizeof(f64) * H2);
  f64 *sb = sig ? sqlite3_malloc(sizeof(f64) * H2) : NULL;
  if (!fb || (sig && !sb)) {
    sqlite3_free(fb);
    sqlite3_free(sb);
    return -SQLITE_NOMEM;
  }
  for (int f = 0; f < folds; f++) {
    int c = c_max - (folds - 1 - f); /* ascending cutoffs; last fold = c_max */
    model(y, c, H2, fb, sb);
    for (int h = 0; h < horizon; h++) {
      int step = gap + h;
      act[f * horizon + h] = y[c + step];
      fcst[f * horizon + h] = fb[step];
      if (sig)
        sig[f * horizon + h] = sb[step];
    }
    if (cutoff_ms)
      cutoff_ms[f] = ts[c - 1];
  }
  sqlite3_free(fb);
  sqlite3_free(sb);
  return folds;
}

/* Pick the TS_MODELS index that minimizes rolling-origin MASE on this series.
 * scale is naive_scale(y,n); act/fcst are scratch [nfolds*horizon]. Returns the
 * index, -1 if no fold was usable, or a negative SQLITE_ code on OOM. */
static int ts_auto_select(const f64 *y, const i64 *ts, int n, int horizon,
                          int gap, int nfolds, f64 scale, f64 *act, f64 *fcst) {
  int best = -1;
  f64 best_mase = 0;
  for (usize mi = 0; mi < countof(TS_MODELS); mi++) {
    int nf = ts_backtest(TS_MODELS[mi].run, y, ts, n, horizon, gap, nfolds, act,
                         fcst, NULL, NULL);
    if (nf < 0)
      return nf;
    if (nf == 0)
      continue;
    f64 mae = metric_mae(act, fcst, nf * horizon);
    f64 mase = scale > 0 ? mae / scale : mae;
    if (best < 0 || mase < best_mase) {
      best = (int)mi;
      best_mase = mase;
    }
  }
  return best;
}

/* Per-step conformal absolute-residual quantiles at level conf, from an out-of-
 * sample rolling-origin backtest of `model`. Fills q[horizon]; act/fcst are
 * scratch [nfolds*horizon] and col is scratch [nfolds]. Returns the fold count
 * used, 0 if too short, or a negative SQLITE_ code on OOM. */
static int ts_conformal_q(predict0_ts_model model, const f64 *y, const i64 *ts,
                          int n, int horizon, int gap, int nfolds, f64 conf,
                          f64 *act, f64 *fcst, f64 *col, f64 *q) {
  int nf = ts_backtest(model, y, ts, n, horizon, gap, nfolds, act, fcst, NULL,
                       NULL);
  if (nf <= 0)
    return nf;
  for (int h = 0; h < horizon; h++) {
    for (int f = 0; f < nf; f++)
      col[f] = fabs(act[f * horizon + h] - fcst[f * horizon + h]);
    for (int i = 1; i < nf; i++) { /* insertion sort (nf small) */
      f64 v = col[i];
      int j = i - 1;
      while (j >= 0 && col[j] > v) {
        col[j + 1] = col[j];
        j--;
      }
      col[j + 1] = v;
    }
    int rank = (int)ceil((f64)(nf + 1) * conf); /* conformal quantile index */
    if (rank > nf)
      rank = nf;
    if (rank < 1)
      rank = 1;
    q[h] = col[rank - 1];
  }
  return nf;
}

/* ---- forecast student serving (PSFCST, §4.1.6) ---- */

/* Forecast H steps from the length-L window `src` (raw units): instance-
 * normalize by the window's own (mean, sd), apply the net, de-normalize. */
static void fcst_infer(const MLP *m, const f64 *src, f32 *win, f32 *hid,
                       f32 *out, f64 *mu_out, f64 *sd_out) {
  int L = m->nfeat;
  f64 mu = 0;
  for (int i = 0; i < L; i++)
    mu += src[i];
  mu /= L;
  f64 v = 0;
  for (int i = 0; i < L; i++) {
    f64 d = src[i] - mu;
    v += d * d;
  }
  f64 sd = sqrt(v / L);
  if (sd < 1e-9)
    sd = 1e-9;
  for (int i = 0; i < L; i++)
    win[i] = (f32)((src[i] - mu) / sd);
  mlp_forward(m, win, hid, out);
  *mu_out = mu;
  *sd_out = sd;
}

/* Value of the quantile at level p by linear interpolation over the student's
 * (ascending) levels, extrapolating the tails (e.g. deciles to a 95% band). */
f64 predict0_quantile_at(const f32 *lev, const f64 *val, int Q, f64 p) {
  if (Q == 1)
    return val[0];
  if (p <= lev[0])
    return val[0] + (val[1] - val[0]) / (lev[1] - lev[0]) * (p - lev[0]);
  if (p >= lev[Q - 1])
    return val[Q - 1] +
           (val[Q - 1] - val[Q - 2]) / (lev[Q - 1] - lev[Q - 2]) * (p - lev[Q - 1]);
  for (int i = 1; i < Q; i++)
    if (p <= lev[i])
      return val[i - 1] +
             (p - lev[i - 1]) / (lev[i] - lev[i - 1]) * (val[i] - val[i - 1]);
  return val[Q - 1];
}

/* Serve a forecast student over series y[0..n-1]. Fills fc/lo/hi[horizon] (the
 * point and the interval at `conf`). For a quantile student (nquant>1) the
 * point and bounds come straight off the emitted fan; for a point student the
 * bound comes from a refit-free backtest of the student over its own history.
 * horizon MUST be <= fs->horizon (checked by the caller). */
static int fcst_run(const ForecastStudent *fs, const f64 *y, int n, int horizon,
                    f64 conf, f64 *fc, f64 *lo, f64 *hi) {
  const MLP *m = &fs->mlp;
  int L = m->nfeat, H = fs->horizon, Q = fs->nquant, rc = SQLITE_OK;
  f32 *win = sqlite3_malloc(sizeof(f32) * L);
  f32 *hid = sqlite3_malloc(sizeof(f32) * (m->nhid > 0 ? m->nhid : 1));
  f32 *out = sqlite3_malloc(sizeof(f32) * m->nout);
  f64 *wbuf = sqlite3_malloc(sizeof(f64) * L);
  f64 *val = sqlite3_malloc(sizeof(f64) * Q);
  f64 *ss = sqlite3_malloc(sizeof(f64) * H);
  int *cnt = sqlite3_malloc(sizeof(int) * H);
  if (!win || !hid || !out || !wbuf || !val || !ss || !cnt) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  /* most-recent window, front-padded with y[0] when the series is short */
  for (int i = 0; i < L; i++) {
    int idx = n - L + i;
    wbuf[i] = idx >= 0 ? y[idx] : y[0];
  }
  f64 mu, sd;
  fcst_infer(m, wbuf, win, hid, out, &mu, &sd);
  f64 plo = (1 - conf) / 2, phi = (1 + conf) / 2;
  for (int h = 0; h < horizon; h++) {
    for (int qq = 0; qq < Q; qq++) /* de-normalize this step's fan */
      val[qq] = (f64)out[h * Q + qq] * sd + mu;
    for (int i = 1; i < Q; i++) /* sort for monotone quantiles */
      for (int j = i; j > 0 && val[j] < val[j - 1]; j--) {
        f64 t = val[j];
        val[j] = val[j - 1];
        val[j - 1] = t;
      }
    fc[h] = predict0_quantile_at(fs->levels, val, Q, 0.5);
    if (Q > 1) {
      lo[h] = predict0_quantile_at(fs->levels, val, Q, plo);
      hi[h] = predict0_quantile_at(fs->levels, val, Q, phi);
    }
  }
  if (Q == 1) { /* point student: interval from a refit-free backtest */
    f64 z = predict0_norm_quantile(0.5 + conf / 2);
    for (int k = 0; k < H; k++) {
      ss[k] = 0;
      cnt[k] = 0;
    }
    int stride = H > 1 ? H / 2 : 1;
    for (int t = L; t + H <= n; t += stride) {
      f64 bmu, bsd;
      fcst_infer(m, y + t - L, win, hid, out, &bmu, &bsd);
      for (int k = 0; k < H; k++) {
        f64 e = y[t + k] - ((f64)out[k] * bsd + bmu);
        ss[k] += e * e;
        cnt[k]++;
      }
    }
    for (int h = 0; h < horizon; h++) {
      f64 sg = cnt[h] ? sqrt(ss[h] / cnt[h]) : sd; /* fallback: window scale */
      lo[h] = fc[h] - z * sg;
      hi[h] = fc[h] + z * sg;
    }
  }
done:
  sqlite3_free(win);
  sqlite3_free(hid);
  sqlite3_free(out);
  sqlite3_free(wbuf);
  sqlite3_free(val);
  sqlite3_free(ss);
  sqlite3_free(cnt);
  return rc;
}

/* Load a forecast student registered under model_id, if the id names one.
 * Returns 1 and fills *fs on hit; 0 if the id is not a registered forecast
 * student; a negative SQLITE_ code (and *errmsg) on a load error. */
static int load_fcst_student(sqlite3 *db, const char *model_id,
                             ForecastStudent *fs, char **errmsg) {
  /* any PSFCST version (the deserializer validates the exact version) */
  static const char FCST_PREFIX[6] = {'P', 'S', 'F', 'C', 'S', 'T'};
  predict0_model_row row;
  int lr = predict0_registry_lookup(db, model_id, &row);
  if (lr != 0)
    return 0; /* absent or lookup error: treat as not-a-forecast-student */
  int is_fcst = row.weights && row.weights_len >= (int)sizeof(FCST_PREFIX) &&
                memcmp(row.weights, FCST_PREFIX, sizeof(FCST_PREFIX)) == 0;
  if (!is_fcst) {
    predict0_model_row_free(&row);
    return 0;
  }
  int rc = fcst_deserialize(row.weights, row.weights_len, fs, errmsg);
  predict0_model_row_free(&row);
  return rc == SQLITE_OK ? 1 : -rc;
}

/* Point forecast of a student over y[0..n): the median of its emitted fan per
 * step (no interval). Lets a student be rolling-origin-evaluated as an auto
 * candidate without the cost of its interval backtest. */
static int fcst_point(const ForecastStudent *fs, const f64 *y, int n,
                      int horizon, f64 *fc) {
  const MLP *m = &fs->mlp;
  int L = m->nfeat, Q = fs->nquant, rc = SQLITE_OK;
  f32 *win = sqlite3_malloc(sizeof(f32) * L);
  f32 *hid = sqlite3_malloc(sizeof(f32) * (m->nhid > 0 ? m->nhid : 1));
  f32 *out = sqlite3_malloc(sizeof(f32) * m->nout);
  f64 *wbuf = sqlite3_malloc(sizeof(f64) * L);
  f64 *val = sqlite3_malloc(sizeof(f64) * Q);
  if (!win || !hid || !out || !wbuf || !val) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  for (int i = 0; i < L; i++) {
    int idx = n - L + i;
    wbuf[i] = idx >= 0 ? y[idx] : y[0];
  }
  f64 mu, sd;
  fcst_infer(m, wbuf, win, hid, out, &mu, &sd);
  for (int h = 0; h < horizon; h++) {
    for (int qq = 0; qq < Q; qq++)
      val[qq] = (f64)out[h * Q + qq] * sd + mu;
    for (int i = 1; i < Q; i++)
      for (int j = i; j > 0 && val[j] < val[j - 1]; j--) {
        f64 t = val[j];
        val[j] = val[j - 1];
        val[j - 1] = t;
      }
    fc[h] = predict0_quantile_at(fs->levels, val, Q, 0.5);
  }
done:
  sqlite3_free(win);
  sqlite3_free(hid);
  sqlite3_free(out);
  sqlite3_free(wbuf);
  sqlite3_free(val);
  return rc;
}

/* An auto candidate: a bundled statistical model or a loaded forecast student. */
typedef struct {
  const char *id;          /* borrowed: static id, or opts.candidates[i] */
  predict0_ts_model stat;  /* non-NULL for a statistical model */
  ForecastStudent student; /* valid when is_student */
  int is_student;
} Candidate;

/* Point forecast of a candidate on y[0..n) -> fc[horizon]. */
static int candidate_point(const Candidate *c, const f64 *y, int n, int horizon,
                           f64 *fc) {
  if (c->is_student)
    return fcst_point(&c->student, y, n, horizon, fc);
  c->stat(y, n, horizon, fc, NULL);
  return SQLITE_OK;
}

/* Rolling-origin aggregate MAE of a candidate (the metric auto ranks by; the
 * MASE scale is constant per series, so MAE ordering == MASE ordering).
 * fc_scratch is sized >= gap+horizon. Sets *nf to folds used (0 if too short);
 * returns the MAE, or a negative SQLITE_ code on OOM. */
static f64 candidate_mae(const Candidate *c, const f64 *y, int n, int horizon,
                         int gap, int nfolds, f64 *fc_scratch, int *nf) {
  int H2 = gap + horizon, c_max = n - H2, c_min = FORECAST_MIN_HISTORY;
  *nf = 0;
  if (c_max < c_min)
    return 0;
  int avail = c_max - c_min + 1;
  int folds = nfolds < avail ? nfolds : avail;
  f64 sum = 0;
  int cnt = 0;
  for (int f = 0; f < folds; f++) {
    int cc = c_max - (folds - 1 - f);
    int rc = candidate_point(c, y, cc, H2, fc_scratch);
    if (rc != SQLITE_OK)
      return -rc;
    for (int h = 0; h < horizon; h++) {
      sum += fabs(y[cc + gap + h] - fc_scratch[gap + h]);
      cnt++;
    }
  }
  *nf = folds;
  return cnt ? sum / cnt : 0;
}

#pragma endregion

#pragma region options

typedef struct {
  char *time_col;
  char *value_col;
  char *group_cols[FORECAST_MAX_GROUP_COLS];
  int n_group_cols;
  f64 confidence;
  int context_limit;
  char *model;
  int receipt;
  int interval_conformal; /* interval_method: 0 residual (default), 1 conformal */
  int folds;              /* rolling-origin folds; 0 = FORECAST_DEFAULT_FOLDS */
  int gap;                /* leakage gap between train end and first target */
  char *candidates[FORECAST_MAX_CANDIDATES]; /* auto pool, when non-empty */
  int n_candidates;
} ForecastOpts;

static void forecast_opts_free(ForecastOpts *o) {
  sqlite3_free(o->time_col);
  sqlite3_free(o->value_col);
  sqlite3_free(o->model);
  for (int i = 0; i < o->n_group_cols; i++)
    sqlite3_free(o->group_cols[i]);
  for (int i = 0; i < o->n_candidates; i++)
    sqlite3_free(o->candidates[i]);
}

static char *dup_text(sqlite3_value *v) {
  const char *t = (const char *)sqlite3_value_text(v);
  return t ? sqlite3_mprintf("%s", t) : NULL;
}

static int forecast_opt_cb(void *ctx, const char *key, sqlite3_value *value,
                           char **errmsg) {
  ForecastOpts *o = ctx;
  int vtype = sqlite3_value_type(value);

  if (strcmp(key, "time_col") == 0 || strcmp(key, "value_col") == 0 ||
      strcmp(key, "model") == 0) {
    if (vtype != SQLITE_TEXT)
      goto wrong_type;
    char **dst = strcmp(key, "time_col") == 0  ? &o->time_col
                 : strcmp(key, "value_col") == 0 ? &o->value_col
                                                 : &o->model;
    sqlite3_free(*dst); /* duplicate key: free before overwrite */
    *dst = dup_text(value);
    return 0;
  }
  if (strcmp(key, "confidence_level") == 0 ||
      strcmp(key, "anomaly_prob_threshold") == 0) {
    if (vtype != SQLITE_FLOAT && vtype != SQLITE_INTEGER)
      goto wrong_type;
    o->confidence = sqlite3_value_double(value);
    if (o->confidence <= 0 || o->confidence >= 1) {
      *errmsg = sqlite3_mprintf(
          "%s: %s must be in (0,1)",
          strcmp(key, "confidence_level") == 0 ? PREDICT_ERR_OPTIONS
                                               : PREDICT_ERR_THRESHOLD,
          key);
      return 1;
    }
    return 0;
  }
  if (strcmp(key, "context_limit") == 0) {
    if (vtype != SQLITE_INTEGER)
      goto wrong_type;
    o->context_limit = sqlite3_value_int(value);
    if (o->context_limit < 1) {
      *errmsg = sqlite3_mprintf("%s: context_limit must be >= 1",
                                PREDICT_ERR_OPTIONS);
      return 1;
    }
    return 0;
  }
  if (strcmp(key, "receipt") == 0) {
    if (vtype != SQLITE_INTEGER)
      goto wrong_type;
    o->receipt = sqlite3_value_int(value) != 0;
    return 0;
  }
  if (strcmp(key, "group_cols") == 0) {
    /* JSON array of names, or a bare string as a one-element array */
    const char *t = (const char *)sqlite3_value_text(value);
    if (!t)
      goto wrong_type;
    /* duplicate key: free any previously-parsed group cols first */
    for (int g = 0; g < o->n_group_cols; g++)
      sqlite3_free(o->group_cols[g]);
    o->n_group_cols = 0;
    if (t[0] != '[') {
      if (vtype != SQLITE_TEXT)
        goto wrong_type;
      o->group_cols[0] = sqlite3_mprintf("%s", t);
      o->n_group_cols = 1;
      return 0;
    }
    /* crude but sufficient: strip [ ] " and split on , — column names
     * with commas or quotes are not supported (and not sane) */
    const char *pch = t + 1;
    while (*pch && *pch != ']') {
      while (*pch == ' ' || *pch == '"' || *pch == ',')
        pch++;
      if (*pch == ']' || !*pch)
        break;
      const char *end = pch;
      while (*end && *end != '"' && *end != ',' && *end != ']')
        end++;
      if (o->n_group_cols >= FORECAST_MAX_GROUP_COLS) {
        *errmsg = sqlite3_mprintf("%s: too many group_cols (max %d)",
                                  PREDICT_ERR_OPTIONS, FORECAST_MAX_GROUP_COLS);
        return 1;
      }
      o->group_cols[o->n_group_cols++] =
          sqlite3_mprintf("%.*s", (int)(end - pch), pch);
      pch = end;
    }
    if (o->n_group_cols == 0) {
      *errmsg = sqlite3_mprintf("%s: group_cols must not be empty",
                                PREDICT_ERR_OPTIONS);
      return 1;
    }
    return 0;
  }
  if (strcmp(key, "interval_method") == 0) {
    if (vtype != SQLITE_TEXT)
      goto wrong_type;
    const char *m = (const char *)sqlite3_value_text(value);
    if (m && strcmp(m, "residual") == 0)
      o->interval_conformal = 0;
    else if (m && strcmp(m, "conformal") == 0)
      o->interval_conformal = 1;
    else {
      *errmsg = sqlite3_mprintf(
          "%s: interval_method must be 'residual' or 'conformal'",
          PREDICT_ERR_OPTIONS);
      return 1;
    }
    return 0;
  }
  if (strcmp(key, "folds") == 0) {
    if (vtype != SQLITE_INTEGER)
      goto wrong_type;
    o->folds = sqlite3_value_int(value);
    if (o->folds < 1 || o->folds > FORECAST_MAX_FOLDS) {
      *errmsg = sqlite3_mprintf("%s: folds must be in 1..%d",
                                PREDICT_ERR_OPTIONS, FORECAST_MAX_FOLDS);
      return 1;
    }
    return 0;
  }
  if (strcmp(key, "gap") == 0) {
    if (vtype != SQLITE_INTEGER)
      goto wrong_type;
    o->gap = sqlite3_value_int(value);
    if (o->gap < 0) {
      *errmsg = sqlite3_mprintf("%s: gap must be >= 0", PREDICT_ERR_OPTIONS);
      return 1;
    }
    return 0;
  }
  if (strcmp(key, "candidates") == 0) {
    /* JSON array of model ids (or a bare name), the pool auto ranks over */
    const char *t = (const char *)sqlite3_value_text(value);
    if (!t)
      goto wrong_type;
    for (int c = 0; c < o->n_candidates; c++)
      sqlite3_free(o->candidates[c]);
    o->n_candidates = 0;
    if (t[0] != '[') {
      if (vtype != SQLITE_TEXT)
        goto wrong_type;
      o->candidates[0] = sqlite3_mprintf("%s", t);
      o->n_candidates = 1;
      return 0;
    }
    const char *pch = t + 1;
    while (*pch && *pch != ']') {
      while (*pch == ' ' || *pch == '"' || *pch == ',')
        pch++;
      if (*pch == ']' || !*pch)
        break;
      const char *end = pch;
      while (*end && *end != '"' && *end != ',' && *end != ']')
        end++;
      if (o->n_candidates >= FORECAST_MAX_CANDIDATES) {
        *errmsg = sqlite3_mprintf("%s: too many candidates (max %d)",
                                  PREDICT_ERR_OPTIONS, FORECAST_MAX_CANDIDATES);
        return 1;
      }
      o->candidates[o->n_candidates++] =
          sqlite3_mprintf("%.*s", (int)(end - pch), pch);
      pch = end;
    }
    if (o->n_candidates == 0) {
      *errmsg = sqlite3_mprintf("%s: candidates must not be empty",
                                PREDICT_ERR_OPTIONS);
      return 1;
    }
    return 0;
  }

wrong_type:
  *errmsg = sqlite3_mprintf("%s: wrong type for option '%s'",
                            PREDICT_ERR_OPTIONS, key);
  return 1;
}

static const char *const FORECAST_OPTION_KEYS[] = {
    "time_col",      "value_col",       "group_cols", "confidence_level",
    "context_limit", "model",           "receipt",    "interval_method",
    "folds",         "gap",             "candidates", NULL};

static const char *const BACKTEST_OPTION_KEYS[] = {
    "time_col",      "value_col", "group_cols",      "confidence_level",
    "context_limit", "folds",     "gap",             "interval_method",
    "model",         "receipt",   NULL};

/* detect_anomalies shares the struct; anomaly_prob_threshold lands in
 * .confidence (same (0,1) validation) with a 0.99 default. */
static const char *const ANOMALY_OPTION_KEYS[] = {
    "time_col", "value_col", "group_cols", "anomaly_prob_threshold",
    "context_limit", "model", "receipt", NULL};

/* Resolve opts->candidates into a Candidate array (statistical models +
 * distilled forecast students). Rejects onnx/unknown ids, a conformal request
 * over a student candidate, and a student trained for a shorter horizon than
 * gap+horizon. On SQLITE_OK the caller owns *out and MUST free each loaded
 * student (fcst_student_free) then the array; on failure everything is freed
 * here and *errmsg is set. */
static int resolve_candidates(sqlite3 *db, const ForecastOpts *opts, int horizon,
                              Candidate **out, int *out_n, char **errmsg) {
  *out = NULL;
  *out_n = 0;
  Candidate *cands = sqlite3_malloc(sizeof(Candidate) * opts->n_candidates);
  if (!cands)
    return SQLITE_NOMEM;
  memset(cands, 0, sizeof(Candidate) * opts->n_candidates);
  int n = 0, rc = SQLITE_OK;
  for (int ci = 0; ci < opts->n_candidates; ci++) {
    const char *id = opts->candidates[ci];
    predict0_ts_model sm = resolve_ts_model(id);
    if (sm) {
      cands[n].id = resolve_ts_model_id(id);
      cands[n].stat = sm;
      cands[n].is_student = 0;
      n++;
      continue;
    }
    char *lerr = NULL;
    ForecastStudent fs;
    int lr = load_fcst_student(db, id, &fs, &lerr);
    sqlite3_free(lerr);
    if (lr == 1) {
      if (opts->interval_conformal) {
        fcst_student_free(&fs);
        *errmsg = sqlite3_mprintf("%s: interval_method='conformal' is not "
                                  "supported for a forecast-student candidate: "
                                  "%s",
                                  PREDICT_ERR_OPTIONS, id);
        rc = SQLITE_ERROR;
        break;
      }
      if (opts->gap + horizon > fs.horizon) {
        fcst_student_free(&fs);
        *errmsg = sqlite3_mprintf("%s: candidate student was trained for a "
                                  "shorter horizon: %s",
                                  PREDICT_ERR_HORIZON, id);
        rc = SQLITE_ERROR;
        break;
      }
      cands[n].id = id; /* borrowed from opts->candidates */
      cands[n].student = fs;
      cands[n].is_student = 1;
      n++;
      continue;
    }
    /* onnx forecast models are valid but too slow to backtest as candidates;
     * unknown ids land here too */
    *errmsg = sqlite3_mprintf("%s: candidate must be a statistical model or a "
                              "distilled forecast student: %s",
                              PREDICT_ERR_MODEL_NOT_FOUND, id);
    rc = SQLITE_ERROR;
    break;
  }
  if (rc != SQLITE_OK) {
    for (int i = 0; i < n; i++)
      if (cands[i].is_student)
        fcst_student_free(&cands[i].student);
    sqlite3_free(cands);
    return rc;
  }
  *out = cands;
  *out_n = n;
  return SQLITE_OK;
}

#pragma endregion

#pragma region vtab

typedef struct {
  char *series_key;
  int step;
  char ts[PREDICT_TS_BUFSIZE];
  f64 forecast, lower, upper;
  const char *status; /* static strings only */
  int has_values;     /* 0 for status-only rows */
  char receipt_id[PREDICT_ULID_BUFSIZE];
} ForecastRow;

static int forecast_row_cmp(const void *a, const void *b) {
  const ForecastRow *x = *(ForecastRow *const *)a;
  const ForecastRow *y = *(ForecastRow *const *)b;
  int c = strcmp(x->series_key ? x->series_key : "",
                 y->series_key ? y->series_key : "");
  if (c)
    return c;
  return x->step - y->step;
}

typedef struct forecast_vtab {
  sqlite3_vtab base;
  sqlite3 *db;
} forecast_vtab;

typedef struct forecast_cursor {
  sqlite3_vtab_cursor base;
  ForecastRow *rows;
  int n_rows;
  int i;
} forecast_cursor;

/* one input series during collection */
typedef struct {
  char *key;
  i64 *ts;
  f64 *val;
  int n, cap;
  int non_numeric;
  int truncated;
} SeriesBuf;

static int fc_connect(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  forecast_vtab *v = sqlite3_malloc(sizeof(*v));
  if (!v)
    return SQLITE_NOMEM;
  memset(v, 0, sizeof(*v));
  v->db = db;
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(series_key TEXT, step INTEGER,"
          " forecast_timestamp TEXT, forecast REAL, lower_bound REAL,"
          " upper_bound REAL, status TEXT, receipt_id TEXT,"
          " query HIDDEN, horizon HIDDEN, options HIDDEN)");
  if (rc != SQLITE_OK) {
    sqlite3_free(v);
    return rc;
  }
  *ppVtab = &v->base;
  return SQLITE_OK;
}

static int fc_disconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int fc_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pIdx) {
  int seen_query = 0, seen_horizon = 0, argv = 1;
  for (int i = 0; i < pIdx->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdx->aConstraint[i];
    if (c->op != SQLITE_INDEX_CONSTRAINT_EQ)
      continue;
    if (c->iColumn == FC_COL_QUERY) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 1;
      pIdx->aConstraintUsage[i].omit = 1;
      seen_query = 1;
    } else if (c->iColumn == FC_COL_HORIZON) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 2;
      pIdx->aConstraintUsage[i].omit = 1;
      seen_horizon = 1;
    } else if (c->iColumn == FC_COL_OPTIONS) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 3;
      pIdx->aConstraintUsage[i].omit = 1;
    }
  }
  UNUSED_PARAMETER(argv);
  if (!seen_query || !seen_horizon) {
    pVtab->zErrMsg = sqlite3_mprintf(
        "%s: forecast(query, horizon) requires both arguments",
        PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  pIdx->estimatedCost = 1000;
  return SQLITE_OK;
}

static int fc_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCur) {
  UNUSED_PARAMETER(pVtab);
  forecast_cursor *c = sqlite3_malloc(sizeof(*c));
  if (!c)
    return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  *ppCur = &c->base;
  return SQLITE_OK;
}

static void fc_rows_free(forecast_cursor *c) {
  for (int i = 0; i < c->n_rows; i++)
    sqlite3_free(c->rows[i].series_key);
  sqlite3_free(c->rows);
  c->rows = NULL;
  c->n_rows = 0;
  c->i = 0;
}

static int fc_close(sqlite3_vtab_cursor *pCur) {
  forecast_cursor *c = (forecast_cursor *)pCur;
  fc_rows_free(c);
  sqlite3_free(c);
  return SQLITE_OK;
}

static int fc_error(forecast_cursor *cur, const char *code, const char *fmt,
                    const char *detail) {
  sqlite3_free(cur->base.pVtab->zErrMsg);
  cur->base.pVtab->zErrMsg =
      sqlite3_mprintf("%s: %s%s", code, fmt, detail ? detail : "");
  return SQLITE_ERROR;
}

static SeriesBuf *series_find(SeriesBuf **all, int *n_series, int *cap,
                              const char *key) {
  for (int i = 0; i < *n_series; i++) {
    if (strcmp((*all)[i].key, key) == 0)
      return &(*all)[i];
  }
  if (*n_series == *cap) {
    int nc = *cap ? *cap * 2 : 8;
    SeriesBuf *grown = sqlite3_realloc(*all, sizeof(SeriesBuf) * nc);
    if (!grown)
      return NULL;
    *all = grown;
    *cap = nc;
  }
  SeriesBuf *s = &(*all)[(*n_series)++];
  memset(s, 0, sizeof(*s));
  s->key = sqlite3_mprintf("%s", key);
  return s;
}

/* sort series rows chronologically (insertion sort: input is usually
 * already ordered, making this near-linear) */
static void series_sort(SeriesBuf *s) {
  for (int i = 1; i < s->n; i++) {
    i64 t = s->ts[i];
    f64 v = s->val[i];
    int j = i - 1;
    while (j >= 0 && s->ts[j] > t) {
      s->ts[j + 1] = s->ts[j];
      s->val[j + 1] = s->val[j];
      j--;
    }
    s->ts[j + 1] = t;
    s->val[j + 1] = v;
  }
}

static void series_array_free(SeriesBuf *series, int n) {
  for (int i = 0; i < n; i++) {
    sqlite3_free(series[i].key);
    sqlite3_free(series[i].ts);
    sqlite3_free(series[i].val);
  }
  sqlite3_free(series);
}

/* Prepare `query`, resolve or infer the time/value/group columns per
 * `opts`, and collect its rows into a series array. Shared verbatim by
 * forecast() and detect_anomalies(); previously copy-pasted, which is
 * why an earlier epoch-ms fix had to be applied twice.
 *
 * On SQLITE_OK: out_series/out_n hold the collected series (free with
 * series_array_free), and out_time/out_value hold the resolved column
 * names (sqlite3_malloc'd, for the receipt). On failure: returns an
 * error code, frees everything internally, and sets errmsg to a
 * "CODE: detail" string (sqlite3_malloc'd) for the caller's zErrMsg. */
static int collect_series(sqlite3 *db, const char *query,
                          const ForecastOpts *opts, SeriesBuf **out_series,
                          int *out_n, char **out_time, char **out_value,
                          char **errmsg) {
  *out_series = NULL;
  *out_n = 0;
  *out_time = NULL;
  *out_value = NULL;

#define COLLECT_FAIL(code, msg, detail)                                       \
  do {                                                                        \
    *errmsg = sqlite3_mprintf("%s: %s%s", (code), (msg),                      \
                              (detail) ? (detail) : "");                      \
    return SQLITE_ERROR;                                                      \
  } while (0)

  sqlite3_stmt *stmt = NULL;
  const char *tail = NULL;
  if (sqlite3_prepare_v2(db, query, -1, &stmt, &tail) != SQLITE_OK || !stmt) {
    char *e = sqlite3_mprintf("%s: query does not parse: %s",
                              PREDICT_ERR_SCHEMA, sqlite3_errmsg(db));
    if (stmt)
      sqlite3_finalize(stmt);
    *errmsg = e;
    return SQLITE_ERROR;
  }
  while (tail && (*tail == ' ' || *tail == '\n' || *tail == ';' ||
                  *tail == '\t'))
    tail++;
  if (tail && *tail != '\0') {
    sqlite3_finalize(stmt);
    COLLECT_FAIL(PREDICT_ERR_SCHEMA, "query must be a single statement", NULL);
  }
  if (!sqlite3_stmt_readonly(stmt)) {
    sqlite3_finalize(stmt);
    COLLECT_FAIL(PREDICT_ERR_QUERY_NOT_READONLY,
                 "query must be a read-only SELECT", NULL);
  }
  int ncol = sqlite3_column_count(stmt);
  if (ncol < 2) {
    sqlite3_finalize(stmt);
    COLLECT_FAIL(PREDICT_ERR_SCHEMA,
                 "query must yield at least a time and a value column", NULL);
  }

  /* resolve named columns */
  int time_idx = -1, value_idx = -1;
  int group_idx[FORECAST_MAX_GROUP_COLS];
  for (int g = 0; g < opts->n_group_cols; g++)
    group_idx[g] = -1;
  for (int i = 0; i < ncol; i++) {
    const char *name = sqlite3_column_name(stmt, i);
    if (opts->time_col && name && strcmp(name, opts->time_col) == 0)
      time_idx = i;
    if (opts->value_col && name && strcmp(name, opts->value_col) == 0)
      value_idx = i;
    for (int g = 0; g < opts->n_group_cols; g++)
      if (name && strcmp(name, opts->group_cols[g]) == 0)
        group_idx[g] = i;
  }
  if (opts->time_col && time_idx < 0) {
    char *d = sqlite3_mprintf("%s", opts->time_col);
    sqlite3_finalize(stmt);
    *errmsg = sqlite3_mprintf("%s: no such time_col: %s", PREDICT_ERR_SCHEMA,
                              d ? d : "");
    sqlite3_free(d);
    return SQLITE_ERROR;
  }
  if (opts->value_col && value_idx < 0) {
    char *d = sqlite3_mprintf("%s", opts->value_col);
    sqlite3_finalize(stmt);
    *errmsg = sqlite3_mprintf("%s: no such value_col: %s", PREDICT_ERR_SCHEMA,
                              d ? d : "");
    sqlite3_free(d);
    return SQLITE_ERROR;
  }
  for (int g = 0; g < opts->n_group_cols; g++) {
    if (group_idx[g] < 0) {
      char *d = sqlite3_mprintf("%s", opts->group_cols[g]);
      sqlite3_finalize(stmt);
      *errmsg = sqlite3_mprintf("%s: no such group col: %s",
                                PREDICT_ERR_SCHEMA, d ? d : "");
      sqlite3_free(d);
      return SQLITE_ERROR;
    }
  }

  SeriesBuf *series = NULL;
  int n_series = 0, series_cap = 0;
  i64 total_rows = 0;
  int inference_done = (time_idx >= 0 && value_idx >= 0);
  int rc;

  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    if (!inference_done) {
      for (int i = 0; time_idx < 0 && i < ncol; i++) {
        int t = sqlite3_column_type(stmt, i);
        if (t == SQLITE_INTEGER) {
          time_idx = i;
        } else if (t == SQLITE_TEXT) {
          i64 ms;
          if (predict0_parse_timestamp(
                  (const char *)sqlite3_column_text(stmt, i), &ms) == 0)
            time_idx = i;
        }
      }
      for (int i = 0; value_idx < 0 && i < ncol; i++) {
        if (i == time_idx)
          continue;
        int in_groups = 0;
        for (int g = 0; g < opts->n_group_cols; g++)
          if (group_idx[g] == i)
            in_groups = 1;
        if (in_groups)
          continue;
        int t = sqlite3_column_type(stmt, i);
        if (t == SQLITE_INTEGER || t == SQLITE_FLOAT)
          value_idx = i;
      }
      if (time_idx < 0 || value_idx < 0) {
        sqlite3_finalize(stmt);
        series_array_free(series, n_series);
        COLLECT_FAIL(PREDICT_ERR_SCHEMA,
                     "could not infer time/value columns", NULL);
      }
      inference_done = 1;
    }

    if (++total_rows > FORECAST_MAX_TOTAL_ROWS) {
      sqlite3_finalize(stmt);
      series_array_free(series, n_series);
      COLLECT_FAIL(PREDICT_ERR_RESOURCE, "too many input rows", NULL);
    }

    /* dynamic key: fixed buffers would truncate long group values and
     * silently merge distinct series */
    char *key = sqlite3_mprintf("%s", "");
    for (int g = 0; key && g < opts->n_group_cols; g++) {
      const char *gv = (const char *)sqlite3_column_text(stmt, group_idx[g]);
      char sep[2] = {PREDICT_FS, 0};
      key = sqlite3_mprintf("%z%s%s", key, g ? sep : "", gv ? gv : "");
    }
    if (!key) {
      rc = SQLITE_NOMEM;
      break;
    }
    SeriesBuf *s = series_find(&series, &n_series, &series_cap, key);
    sqlite3_free(key);
    if (!s) {
      rc = SQLITE_NOMEM;
      break;
    }
    if (s->non_numeric)
      continue;

    i64 ms = 0;
    int tt = sqlite3_column_type(stmt, time_idx);
    if (tt == SQLITE_INTEGER) {
      /* epoch column: reject negatives and out-of-domain values as
       * non_numeric rather than formatting garbage (the original bug) */
      i64 raw = sqlite3_column_int64(stmt, time_idx);
      if (raw < 0) {
        s->non_numeric = 1;
        continue;
      }
      ms = raw >= PREDICT_EPOCH_MS_THRESHOLD ? raw : raw * 1000;
      if (ms > PREDICT_MS_MAX) {
        s->non_numeric = 1;
        continue;
      }
    } else if (predict0_parse_timestamp(
                   (const char *)sqlite3_column_text(stmt, time_idx), &ms)) {
      s->non_numeric = 1;
      continue;
    }
    int vt = sqlite3_column_type(stmt, value_idx);
    if (vt != SQLITE_INTEGER && vt != SQLITE_FLOAT) {
      s->non_numeric = 1;
      continue;
    }

    if (s->n == s->cap) {
      int nc = s->cap ? s->cap * 2 : 64;
      i64 *nts = sqlite3_realloc(s->ts, sizeof(i64) * nc);
      f64 *nval = sqlite3_realloc(s->val, sizeof(f64) * nc);
      if (!nts || !nval) {
        sqlite3_free(nts);
        sqlite3_free(nval);
        rc = SQLITE_NOMEM;
        break;
      }
      s->ts = nts;
      s->val = nval;
      s->cap = nc;
    }
    s->ts[s->n] = ms;
    s->val[s->n] = sqlite3_column_double(stmt, value_idx);
    s->n++;
  }

  char *rtime = NULL, *rvalue = NULL;
  if (time_idx >= 0)
    rtime = sqlite3_mprintf("%s", sqlite3_column_name(stmt, time_idx));
  if (value_idx >= 0)
    rvalue = sqlite3_mprintf("%s", sqlite3_column_name(stmt, value_idx));
  sqlite3_finalize(stmt);

  if (rc == SQLITE_NOMEM) {
    sqlite3_free(rtime);
    sqlite3_free(rvalue);
    series_array_free(series, n_series);
    return SQLITE_NOMEM;
  }
  if (rc != SQLITE_DONE) {
    sqlite3_free(rtime);
    sqlite3_free(rvalue);
    series_array_free(series, n_series);
    COLLECT_FAIL(PREDICT_ERR_RESOURCE, "query execution failed", NULL);
  }

  *out_series = series;
  *out_n = n_series;
  *out_time = rtime;
  *out_value = rvalue;
  return SQLITE_OK;
#undef COLLECT_FAIL
}

/* Resolved backing-model context for one forecast call. Shared by the
 * query form (fc_filter) and the aggregate form; resolution, serving,
 * hashing, and params-building below are the one implementation of the
 * pipeline both forms run after their rows are collected. */
typedef struct {
  int is_auto;
  predict0_ts_model model; /* NULL for auto/student/onnx */
  ForecastStudent fstudent;
  int have_fstudent;
  int have_onnx;
#ifdef SQLITE_PREDICT_ONNX
  predict0_model_row onnx_row;
#endif
  Candidate *cands;
  int n_cands;
} FcModels;

static void fc_models_free(FcModels *mc) {
  for (int ci = 0; ci < mc->n_cands; ci++)
    if (mc->cands[ci].is_student)
      fcst_student_free(&mc->cands[ci].student);
  sqlite3_free(mc->cands);
  if (mc->have_fstudent)
    fcst_student_free(&mc->fstudent);
#ifdef SQLITE_PREDICT_ONNX
  if (mc->have_onnx)
    predict0_model_row_free(&mc->onnx_row);
#endif
  memset(mc, 0, sizeof(*mc));
}

/* Resolve opts->model (+ any candidates pool) into an FcModels. On
 * failure frees partial state and sets *errmsg ("CODE: detail"). */
static int fc_resolve_models(sqlite3 *db, const ForecastOpts *opts,
                             int horizon, FcModels *mc, char **errmsg) {
  memset(mc, 0, sizeof(*mc));
  mc->is_auto = opts->model && strcmp(opts->model, "auto") == 0;
  mc->model = mc->is_auto ? NULL : resolve_ts_model(opts->model);
  if (!mc->is_auto && !mc->model) {
    /* not a bundled model: maybe a registered forecast student (PSFCST) */
    char *lerr = NULL;
    int lr = load_fcst_student(db, opts->model, &mc->fstudent, &lerr);
    sqlite3_free(lerr);
    if (lr < 0) {
      *errmsg = sqlite3_mprintf("%s: invalid forecast student: %s",
                                PREDICT_ERR_SCHEMA, opts->model);
      return SQLITE_ERROR;
    }
    if (lr == 1) {
      mc->have_fstudent = 1;
      if (horizon > mc->fstudent.horizon) {
        fc_models_free(mc);
        *errmsg = sqlite3_mprintf(
            "%s: horizon exceeds the forecast student's trained horizon",
            PREDICT_ERR_HORIZON);
        return SQLITE_ERROR;
      }
    } else { /* not a native student: maybe a registered onnx forecast model */
      predict0_model_row row;
      int rl = predict0_registry_lookup(db, opts->model, &row);
      int is_onnx = rl == 0 && row.runtime && strcmp(row.runtime, "onnx") == 0;
      if (!is_onnx) {
        if (rl == 0)
          predict0_model_row_free(&row);
        *errmsg = sqlite3_mprintf("%s: no such model: %s",
                                  PREDICT_ERR_MODEL_NOT_FOUND, opts->model);
        return SQLITE_ERROR;
      }
#ifdef SQLITE_PREDICT_ONNX
      mc->onnx_row = row; /* ownership; freed by fc_models_free */
      mc->have_onnx = 1;
#else
      predict0_model_row_free(&row);
      *errmsg =
          sqlite3_mprintf("%s: onnx runtime is not in this build: %s",
                          PREDICT_ERR_RUNTIME_UNAVAILABLE, opts->model);
      return SQLITE_ERROR;
#endif
    }
  }

  if (opts->interval_conformal && (mc->have_fstudent || mc->have_onnx)) {
    fc_models_free(mc);
    *errmsg = sqlite3_mprintf(
        "%s: interval_method='conformal' is only supported for the "
        "statistical models (theta-classic, stub-seasonal-naive, auto)",
        PREDICT_ERR_OPTIONS);
    return SQLITE_ERROR;
  }

  /* auto over an explicit candidate set: resolve the statistical models +
   * distilled forecast students (validated) */
  if (opts->n_candidates > 0) {
    if (!mc->is_auto) {
      fc_models_free(mc);
      *errmsg = sqlite3_mprintf("%s: candidates requires model=\"auto\"",
                                PREDICT_ERR_OPTIONS);
      return SQLITE_ERROR;
    }
    int crc =
        resolve_candidates(db, opts, horizon, &mc->cands, &mc->n_cands, errmsg);
    if (crc != SQLITE_OK) {
      mc->n_cands = 0;
      mc->cands = NULL;
      fc_models_free(mc);
      return crc;
    }
  }
  return SQLITE_OK;
}

/* the receipt's model_id for a resolved forecast call */
static const char *fc_model_id(const ForecastOpts *opts, const FcModels *mc) {
  if (mc->is_auto)
    return "auto";
  if (mc->have_fstudent || mc->have_onnx)
    return opts->model;
  return resolve_ts_model_id(opts->model);
}

/* Forecast every collected series into ForecastRow output rows
 * (sqlite3_malloc'd into *rows_out). Returns SQLITE_OK, SQLITE_NOMEM,
 * or an error with *serve_err set ("CODE: detail", a registered model
 * failing mid-serve); on any failure the partial rows are freed here
 * and *rows_out is NULL. */
static int fc_run_series(sqlite3 *db, SeriesBuf *series, int n_series,
                         int horizon, const ForecastOpts *opts, FcModels *mc,
                         ForecastRow **rows_out, int *n_rows_out,
                         char **serve_err) {
  UNUSED_PARAMETER(db); /* used by the onnx serving branch only */
  *rows_out = NULL;
  *n_rows_out = 0;
  *serve_err = NULL;
  int rc = SQLITE_OK;

  f64 z = predict0_norm_quantile(0.5 + opts->confidence / 2);
  int max_rows = 0;
  for (int i = 0; i < n_series; i++)
    max_rows += series[i].non_numeric || series[i].n < FORECAST_MIN_HISTORY
                    ? 1
                    : horizon;

  ForecastRow *rows =
      sqlite3_malloc(sizeof(ForecastRow) * (max_rows ? max_rows : 1));
  int n_rows = 0;
  if (!rows)
    rc = SQLITE_NOMEM;

  f64 *fc = sqlite3_malloc(sizeof(f64) * horizon);
  f64 *sg = sqlite3_malloc(sizeof(f64) * horizon);
  f64 *lo = sqlite3_malloc(sizeof(f64) * horizon);
  f64 *hi = sqlite3_malloc(sizeof(f64) * horizon);

  /* rolling-origin scratch: bt_* for default auto-selection and conformal,
   * cand_fc for candidate-set selection */
  int folds_req = opts->folds > 0 ? opts->folds : FORECAST_DEFAULT_FOLDS;
  int need_bt =
      (mc->is_auto && opts->n_candidates == 0) || opts->interval_conformal;
  f64 *bt_act = NULL, *bt_fcst = NULL, *bt_col = NULL, *bt_q = NULL;
  if (need_bt) {
    bt_act = sqlite3_malloc(sizeof(f64) * (size_t)folds_req * horizon);
    bt_fcst = sqlite3_malloc(sizeof(f64) * (size_t)folds_req * horizon);
    bt_col = sqlite3_malloc(sizeof(f64) * folds_req);
    bt_q = sqlite3_malloc(sizeof(f64) * horizon);
    if (!bt_act || !bt_fcst || !bt_col || !bt_q)
      rc = SQLITE_NOMEM;
  }
  f64 *cand_fc = NULL;
  if (mc->n_cands > 0) {
    cand_fc = sqlite3_malloc(sizeof(f64) * ((size_t)opts->gap + horizon));
    if (!cand_fc)
      rc = SQLITE_NOMEM;
  }

  for (int i = 0; rows && fc && sg && lo && hi &&
                  (!need_bt || (bt_act && bt_fcst && bt_col && bt_q)) &&
                  (mc->n_cands == 0 || cand_fc) && i < n_series;
       i++) {
    SeriesBuf *s = &series[i];
    const char *status = NULL;
    if (s->non_numeric)
      status = "non_numeric";
    else if (s->n < FORECAST_MIN_HISTORY)
      status = "insufficient_history";

    if (status) {
      ForecastRow *r = &rows[n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      r->status = status;
      continue;
    }

    series_sort(s);
    int truncated = 0;
    f64 *y = s->val;
    i64 *ts = s->ts;
    int n = s->n;
    if (opts->context_limit > 0 && n > opts->context_limit) {
      y += n - opts->context_limit;
      ts += n - opts->context_limit;
      n = opts->context_limit;
      truncated = 1;
    }

    /* median step from sorted timestamps */
    i64 step_ms = PREDICT_MS_PER_HOUR;
    if (n > 1) {
      i64 span = ts[n - 1] - ts[0];
      step_ms = span > 0 ? span / (n - 1) : PREDICT_MS_PER_HOUR;
    }

    if (mc->have_fstudent) {
      if (fcst_run(&mc->fstudent, y, n, horizon, opts->confidence, fc, lo,
                   hi) != SQLITE_OK)
        rc = SQLITE_NOMEM;
    } else if (mc->have_onnx) {
#ifdef SQLITE_PREDICT_ONNX
      predict0_backend_opts bopts = {NULL, NULL, NULL, 0};
      int orc =
          predict0_onnx_forecast(db, &mc->onnx_row, &bopts, y, n, horizon,
                                 opts->confidence, fc, lo, hi, serve_err);
      if (orc != SQLITE_OK) {
        rc = orc;
        break; /* serve_err carries the message */
      }
#endif
    } else {
      predict0_ts_model use_model = mc->model;
      const ForecastStudent *win_student = NULL;
      if (mc->is_auto && mc->n_cands > 0) {
        /* rank the explicit candidate set (stat models + students) per series */
        int best = -1;
        f64 best_mae = 0;
        for (int ci = 0; ci < mc->n_cands; ci++) {
          int cnf = 0;
          f64 mae = candidate_mae(&mc->cands[ci], y, n, horizon, opts->gap,
                                  folds_req, cand_fc, &cnf);
          if (mae < 0) {
            rc = SQLITE_NOMEM;
            break;
          }
          if (cnf == 0)
            continue; /* candidate could not be backtested on this series */
          if (best < 0 || mae < best_mae) {
            best = ci;
            best_mae = mae;
          }
        }
        if (rc != SQLITE_OK)
          break;
        if (best < 0) { /* no candidate had a usable fold */
          ForecastRow *r = &rows[n_rows++];
          memset(r, 0, sizeof(*r));
          r->series_key = sqlite3_mprintf("%s", s->key);
          r->status = "insufficient_history";
          continue;
        }
        if (mc->cands[best].is_student)
          win_student = &mc->cands[best].student;
        else
          use_model = mc->cands[best].stat;
      } else if (mc->is_auto) {
        int bi = ts_auto_select(y, ts, n, horizon, opts->gap, folds_req,
                                naive_scale(y, n), bt_act, bt_fcst);
        if (bi <= -2) { /* negative SQLITE code (OOM) */
          rc = SQLITE_NOMEM;
          break;
        }
        /* bi == -1: series too short to backtest -> default to theta */
        use_model = bi >= 0 ? TS_MODELS[bi].run : model_theta;
      }
      if (win_student) {
        if (fcst_run(win_student, y, n, horizon, opts->confidence, fc, lo,
                     hi) != SQLITE_OK) {
          rc = SQLITE_NOMEM;
          break;
        }
      } else if (opts->interval_conformal) {
        int nf = ts_conformal_q(use_model, y, ts, n, horizon, opts->gap,
                                folds_req, opts->confidence, bt_act, bt_fcst,
                                bt_col, bt_q);
        if (nf <= -2) {
          rc = SQLITE_NOMEM;
          break;
        }
        if (nf < CONFORMAL_MIN_FOLDS) { /* cannot calibrate honestly */
          ForecastRow *r = &rows[n_rows++];
          memset(r, 0, sizeof(*r));
          r->series_key = sqlite3_mprintf("%s", s->key);
          r->status = "insufficient_history";
          continue;
        }
        use_model(y, n, horizon, fc, NULL); /* point-only */
        for (int h = 0; h < horizon; h++) {
          lo[h] = fc[h] - bt_q[h];
          hi[h] = fc[h] + bt_q[h];
        }
      } else {
        use_model(y, n, horizon, fc, sg);
        for (int h = 0; h < horizon; h++) { /* symmetric Gaussian interval */
          lo[h] = fc[h] - z * sg[h];
          hi[h] = fc[h] + z * sg[h];
        }
      }
    }
    for (int h = 1; h <= horizon; h++) {
      ForecastRow *r = &rows[n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      r->step = h;
      predict0_format_timestamp(ts[n - 1] + step_ms * h, r->ts, sizeof(r->ts));
      r->forecast = fc[h - 1];
      r->lower = lo[h - 1];
      r->upper = hi[h - 1];
      r->status = truncated ? "truncated" : "ok";
      r->has_values = 1;
    }
  }

  sqlite3_free(fc);
  sqlite3_free(sg);
  sqlite3_free(lo);
  sqlite3_free(hi);
  sqlite3_free(bt_act);
  sqlite3_free(bt_fcst);
  sqlite3_free(bt_col);
  sqlite3_free(bt_q);
  sqlite3_free(cand_fc);

  if (rc != SQLITE_OK || *serve_err) {
    for (int i = 0; i < n_rows; i++)
      sqlite3_free(rows[i].series_key);
    sqlite3_free(rows);
    return rc == SQLITE_OK ? SQLITE_ERROR : rc;
  }
  *rows_out = rows;
  *n_rows_out = n_rows;
  return SQLITE_OK;
}

/* RFC §4.1.3 result hash over rows sorted by (series_key, step).
 * Returns SQLITE_OK or SQLITE_NOMEM. */
static int fc_result_hash(ForecastRow *rows, int n_rows,
                          char out[PREDICT_HEX_BUFSIZE]) {
  ForecastRow **order =
      sqlite3_malloc(sizeof(ForecastRow *) * (n_rows ? n_rows : 1));
  if (!order)
    return SQLITE_NOMEM;
  for (int i = 0; i < n_rows; i++)
    order[i] = &rows[i];
  qsort(order, (usize)n_rows, sizeof(ForecastRow *), forecast_row_cmp);
  predict0_hasher h;
  predict0_hash_init(&h);
  for (int i = 0; i < n_rows; i++) {
    ForecastRow *r = order[i];
    predict0_hash_text(&h, r->series_key);
    if (r->has_values) {
      predict0_hash_int(&h, r->step);
      predict0_hash_text(&h, r->ts);
      predict0_hash_real(&h, r->forecast);
      predict0_hash_real(&h, r->lower);
      predict0_hash_real(&h, r->upper);
    } else {
      for (int k = 0; k < 5; k++)
        predict0_hash_null(&h);
    }
    predict0_hash_row_end(&h);
  }
  sqlite3_free(order);
  predict0_hash_hex(&h, out);
  return SQLITE_OK;
}

/* Canonical receipt params via json_object (keys alphabetical).
 * agg_form omits the query-shape keys (time_col/value_col/group_cols),
 * which do not exist in the aggregate form (RFC §4.2.8). Returns a
 * sqlite3_malloc'd JSON string or NULL. */
static char *fc_build_params(sqlite3 *db, const ForecastOpts *opts,
                             int horizon, const char *model_id,
                             const char *time_col, const char *value_col,
                             int agg_form) {
  char group_json[600] = "";
  if (opts->n_group_cols) {
    usize gl = 0;
    gl += (usize)snprintf(group_json + gl, sizeof(group_json) - gl, "[");
    for (int g = 0; g < opts->n_group_cols && gl < sizeof(group_json) - 4; g++)
      gl += (usize)snprintf(group_json + gl, sizeof(group_json) - gl,
                            "%s\"%s\"", g ? "," : "", opts->group_cols[g]);
    snprintf(group_json + gl, sizeof(group_json) - gl, "]");
  }
  char cand_json[600] = "";
  if (opts->n_candidates) {
    usize cl = 0;
    cl += (usize)snprintf(cand_json + cl, sizeof(cand_json) - cl, "[");
    for (int g = 0; g < opts->n_candidates && cl < sizeof(cand_json) - 4; g++)
      cl += (usize)snprintf(cand_json + cl, sizeof(cand_json) - cl, "%s\"%s\"",
                            g ? "," : "", opts->candidates[g]);
    snprintf(cand_json + cl, sizeof(cand_json) - cl, "]");
  }
  sqlite3_stmt *pj = NULL;
  char *params = NULL;
  const char *sql =
      agg_form
          ? "SELECT json_object("
            " 'candidates', CASE WHEN ?11 = '' THEN NULL ELSE json(?11) END,"
            " 'confidence_level', ?1, 'context_limit', ?2,"
            " 'folds', ?3, 'gap', ?4,"
            " 'horizon', ?6, 'interval_method', ?7, 'model', ?8,"
            " 'receipt', 1)"
          : "SELECT json_object("
            " 'candidates', CASE WHEN ?11 = '' THEN NULL ELSE json(?11) END,"
            " 'confidence_level', ?1, 'context_limit', ?2,"
            " 'folds', ?3, 'gap', ?4,"
            " 'group_cols', CASE WHEN ?5 = '' THEN NULL ELSE json(?5) END,"
            " 'horizon', ?6, 'interval_method', ?7, 'model', ?8,"
            " 'receipt', 1, 'time_col', ?9, 'value_col', ?10)";
  if (sqlite3_prepare_v2(db, sql, -1, &pj, NULL) == SQLITE_OK) {
    sqlite3_bind_double(pj, 1, opts->confidence);
    if (opts->context_limit > 0)
      sqlite3_bind_int(pj, 2, opts->context_limit);
    else
      sqlite3_bind_null(pj, 2);
    if (opts->folds > 0)
      sqlite3_bind_int(pj, 3, opts->folds);
    else
      sqlite3_bind_null(pj, 3);
    if (opts->gap > 0)
      sqlite3_bind_int(pj, 4, opts->gap);
    else
      sqlite3_bind_null(pj, 4);
    if (!agg_form)
      sqlite3_bind_text(pj, 5, group_json, -1, SQLITE_STATIC);
    sqlite3_bind_int(pj, 6, horizon);
    sqlite3_bind_text(pj, 7,
                      opts->interval_conformal ? "conformal" : "residual", -1,
                      SQLITE_STATIC);
    sqlite3_bind_text(pj, 8, model_id, -1, SQLITE_STATIC);
    if (!agg_form) {
      if (time_col)
        sqlite3_bind_text(pj, 9, time_col, -1, SQLITE_STATIC);
      if (value_col)
        sqlite3_bind_text(pj, 10, value_col, -1, SQLITE_STATIC);
    }
    sqlite3_bind_text(pj, 11, cand_json, -1, SQLITE_STATIC);
    if (sqlite3_step(pj) == SQLITE_ROW)
      params = sqlite3_mprintf("%s", (const char *)sqlite3_column_text(pj, 0));
    sqlite3_finalize(pj);
  }
  return params;
}

static int fc_filter(sqlite3_vtab_cursor *pCur, int idxNum,
                     const char *idxStr, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  forecast_cursor *cur = (forecast_cursor *)pCur;
  forecast_vtab *vtab = (forecast_vtab *)cur->base.pVtab;
  sqlite3 *db = vtab->db;
  fc_rows_free(cur);

  if (argc < 2)
    return fc_error(cur, PREDICT_ERR_SCHEMA,
                    "forecast(query, horizon) requires both arguments", NULL);

  const char *query = (const char *)sqlite3_value_text(argv[0]);
  if (!query)
    return fc_error(cur, PREDICT_ERR_SCHEMA, "query must be text", NULL);

  if (sqlite3_value_type(argv[1]) != SQLITE_INTEGER)
    return fc_error(cur, PREDICT_ERR_HORIZON, "horizon must be an integer",
                    NULL);
  int horizon = sqlite3_value_int(argv[1]);
  if (horizon < 1 || horizon > FORECAST_MAX_HORIZON)
    return fc_error(cur, PREDICT_ERR_HORIZON,
                    "horizon out of range 1..1000", NULL);

  ForecastOpts opts;
  memset(&opts, 0, sizeof(opts));
  opts.confidence = 0.95;
  opts.context_limit = FORECAST_DEFAULT_CONTEXT_LIMIT;
  opts.receipt = 1;

  const char *options_json =
      argc >= 3 ? (const char *)sqlite3_value_text(argv[2]) : NULL;
  char *errmsg = NULL;
  if (predict0_options_parse(db, options_json, FORECAST_OPTION_KEYS,
                             forecast_opt_cb, &opts, &errmsg)) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = errmsg;
    forecast_opts_free(&opts);
    return SQLITE_ERROR;
  }

  FcModels mc;
  char *mrerr = NULL;
  if (fc_resolve_models(db, &opts, horizon, &mc, &mrerr) != SQLITE_OK) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = mrerr;
    forecast_opts_free(&opts);
    return SQLITE_ERROR;
  }

  /* prepare + collect the series (shared with detect_anomalies) */
  SeriesBuf *series = NULL;
  int n_series = 0;
  char *resolved_time = NULL, *resolved_value = NULL;
  char *cerr = NULL;
  int rc = collect_series(db, query, &opts, &series, &n_series,
                          &resolved_time, &resolved_value, &cerr);
  if (rc != SQLITE_OK) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = cerr;
    fc_models_free(&mc);
    forecast_opts_free(&opts);
    return rc;
  }

  /* forecast every series into cursor rows; model_id is resolved before
   * the models are freed (it points at a static id or opts.model) */
  const char *model_id = fc_model_id(&opts, &mc);
  char *serve_err = NULL;
  rc = fc_run_series(db, series, n_series, horizon, &opts, &mc, &cur->rows,
                     &cur->n_rows, &serve_err);
  fc_models_free(&mc);
  series_array_free(series, n_series);

  if (rc != SQLITE_OK) {
    if (serve_err) { /* a registered model failed mid-serve */
      sqlite3_free(vtab->base.zErrMsg);
      vtab->base.zErrMsg = serve_err;
    }
    sqlite3_free(resolved_time);
    sqlite3_free(resolved_value);
    forecast_opts_free(&opts);
    return rc;
  }

  /* receipt: RFC §4.1.2 — digest BEFORE insert; params canonical via
   * json_object with alphabetical keys, resolved column names recorded */
  if (opts.receipt) {
    char *rerr = NULL;
    char result_hash[PREDICT_HEX_BUFSIZE];
    if (fc_result_hash(cur->rows, cur->n_rows, result_hash) != SQLITE_OK)
      goto receipt_fail;

    char *params = fc_build_params(db, &opts, horizon, model_id,
                                   resolved_time, resolved_value, 0);
    if (!params) {
      vtab->base.zErrMsg = sqlite3_mprintf(
          "%s: could not canonicalize params", PREDICT_ERR_RESOURCE);
      goto receipt_fail;
    }

    char receipt_id[PREDICT_ULID_BUFSIZE];
    int irc = predict0_emit_receipt(db, "forecast", model_id, params, query,
                                    result_hash, receipt_id, &rerr);
    sqlite3_free(params);
    if (irc != SQLITE_OK) {
      sqlite3_free(vtab->base.zErrMsg);
      vtab->base.zErrMsg = rerr;
      goto receipt_fail;
    }
    for (int i = 0; i < cur->n_rows; i++)
      memcpy(cur->rows[i].receipt_id, receipt_id, sizeof(receipt_id));
  }

  sqlite3_free(resolved_time);
  sqlite3_free(resolved_value);
  forecast_opts_free(&opts);
  cur->i = 0;
  return SQLITE_OK;

receipt_fail:
  sqlite3_free(resolved_time);
  sqlite3_free(resolved_value);
  forecast_opts_free(&opts);
  fc_rows_free(cur);
  return SQLITE_ERROR;
}

static int fc_next(sqlite3_vtab_cursor *pCur) {
  ((forecast_cursor *)pCur)->i++;
  return SQLITE_OK;
}

static int fc_eof(sqlite3_vtab_cursor *pCur) {
  forecast_cursor *c = (forecast_cursor *)pCur;
  return c->i >= c->n_rows;
}

static int fc_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx,
                     int col) {
  forecast_cursor *c = (forecast_cursor *)pCur;
  ForecastRow *r = &c->rows[c->i];
  switch (col) {
  case FC_COL_SERIES_KEY:
    sqlite3_result_text(ctx, r->series_key ? r->series_key : "", -1,
                        SQLITE_TRANSIENT);
    break;
  case FC_COL_STEP:
    if (r->has_values)
      sqlite3_result_int(ctx, r->step);
    break;
  case FC_COL_TS:
    if (r->has_values)
      sqlite3_result_text(ctx, r->ts, -1, SQLITE_TRANSIENT);
    break;
  case FC_COL_FORECAST:
    if (r->has_values)
      sqlite3_result_double(ctx, r->forecast);
    break;
  case FC_COL_LOWER:
    if (r->has_values)
      sqlite3_result_double(ctx, r->lower);
    break;
  case FC_COL_UPPER:
    if (r->has_values)
      sqlite3_result_double(ctx, r->upper);
    break;
  case FC_COL_STATUS:
    sqlite3_result_text(ctx, r->status, -1, SQLITE_STATIC);
    break;
  case FC_COL_RECEIPT:
    if (r->receipt_id[0])
      sqlite3_result_text(ctx, r->receipt_id, -1, SQLITE_TRANSIENT);
    break;
  default:
    break;
  }
  return SQLITE_OK;
}

static int fc_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  *pRowid = ((forecast_cursor *)pCur)->i;
  return SQLITE_OK;
}

static sqlite3_module forecastModule = {
    /* iVersion    */ 0,
    /* xCreate     */ NULL, /* eponymous only */
    /* xConnect    */ fc_connect,
    /* xBestIndex  */ fc_best_index,
    /* xDisconnect */ fc_disconnect,
    /* xDestroy    */ NULL,
    /* xOpen       */ fc_open,
    /* xClose      */ fc_close,
    /* xFilter     */ fc_filter,
    /* xNext       */ fc_next,
    /* xEof        */ fc_eof,
    /* xColumn     */ fc_column,
    /* xRowid      */ fc_rowid,
    /* xUpdate     */ NULL,
    /* xBegin      */ NULL,
    /* xSync       */ NULL,
    /* xCommit     */ NULL,
    /* xRollback   */ NULL,
    /* xFindMethod */ NULL,
    /* xRename     */ NULL,
    /* xSavepoint  */ NULL,
    /* xRelease    */ NULL,
    /* xRollbackTo */ NULL,
    /* xShadowName */ NULL,
    /* xIntegrity  */ NULL};

#pragma endregion

#pragma region subpca

/* Sub-PCA anomaly scorer (RFC §4.2.2, the `sub-pca` model). A subsequence
 * detector, not a forecaster: it embeds the series into sliding windows of one
 * period, standardizes each phase, fits PCA over the windows, and scores each
 * window by its reconstruction error in the top-variance subspace -- a window
 * that does not lie on the "normal" manifold reconstructs poorly. This is the
 * method family that leads the TSB-AD-U benchmark (subsequence reconstruction /
 * clustering), where one-step-residual detectors sit mid-pack. Deterministic:
 * fixed window, fixed component fraction, a deterministic Jacobi eigensolver. */

#define SUBPCA_DEFAULT_WIN 125 /* find_length_rank's fallback when no clear period */
#define SUBPCA_MIN_WIN 6
#define SUBPCA_MAX_WIN 303
#define SUBPCA_KEEP 0.30 /* fraction of top components kept (benchmark-chosen) */

/* Cyclic Jacobi eigensolver for a symmetric w x w matrix A (row-major, over-
 * written). On return d[j] = eigenvalues and V (row-major) holds eigenvectors
 * in columns: V[i*w + j] is coordinate i of eigenvector j. Deterministic (fixed
 * sweep order and iteration cap), so the score replays exactly. */
static void jacobi_eigen(f64 *A, int w, f64 *d, f64 *V) {
  for (int i = 0; i < w; i++) {
    for (int j = 0; j < w; j++)
      V[i * w + j] = (i == j) ? 1.0 : 0.0;
    d[i] = A[i * w + i];
  }
  for (int sweep = 0; sweep < 60; sweep++) {
    f64 off = 0;
    for (int p = 0; p < w; p++)
      for (int q = p + 1; q < w; q++)
        off += A[p * w + q] * A[p * w + q];
    if (off < 1e-30)
      break;
    for (int p = 0; p < w; p++) {
      for (int q = p + 1; q < w; q++) {
        f64 apq = A[p * w + q];
        if (fabs(apq) < 1e-300)
          continue;
        f64 app = A[p * w + p], aqq = A[q * w + q];
        f64 phi = 0.5 * atan2(2.0 * apq, aqq - app);
        f64 c = cos(phi), s = sin(phi);
        for (int i = 0; i < w; i++) { /* rotate rows/cols p,q of A */
          f64 aip = A[i * w + p], aiq = A[i * w + q];
          A[i * w + p] = c * aip - s * aiq;
          A[i * w + q] = s * aip + c * aiq;
        }
        for (int i = 0; i < w; i++) {
          f64 api = A[p * w + i], aqi = A[q * w + i];
          A[p * w + i] = c * api - s * aqi;
          A[q * w + i] = s * api + c * aqi;
        }
        for (int i = 0; i < w; i++) { /* accumulate eigenvectors */
          f64 vip = V[i * w + p], viq = V[i * w + q];
          V[i * w + p] = c * vip - s * viq;
          V[i * w + q] = s * vip + c * viq;
        }
      }
    }
  }
  for (int j = 0; j < w; j++)
    d[j] = A[j * w + j];
}

/* Fill score[0..n) with the Sub-PCA per-point anomaly score; *win_out gets the
 * window used. Returns SQLITE_OK, or SQLITE_NOMEM / SQLITE_ERROR (too short).
 * O(m*w^2) time, O(w^2) memory (the covariance and eigenvectors); the window
 * matrix is never materialized -- each window is recomputed from y. */
static int subpca_score(const f64 *y, int n, int *win_out, f64 *score,
                        char **errmsg) {
  int w = detect_period(y, n);
  if (w < SUBPCA_MIN_WIN || w > SUBPCA_MAX_WIN)
    w = SUBPCA_DEFAULT_WIN;
  if (w > n / 2)
    w = n / 2;
  *win_out = w;
  int m = n - w + 1;
  if (w < 2 || m < w + 1) {
    *errmsg = sqlite3_mprintf("%s: series too short for sub-pca (need > %d)",
                              PREDICT_ERR_SCHEMA, 2 * w);
    return SQLITE_ERROR;
  }
  int k = (int)(SUBPCA_KEEP * w + 0.5);
  if (k < 1)
    k = 1;
  if (k > w)
    k = w;
  int rc = SQLITE_OK;
  f64 *mu = sqlite3_malloc(sizeof(f64) * w);
  f64 *sd = sqlite3_malloc(sizeof(f64) * w);
  f64 *C = sqlite3_malloc(sizeof(f64) * (size_t)w * w);
  f64 *V = sqlite3_malloc(sizeof(f64) * (size_t)w * w);
  f64 *d = sqlite3_malloc(sizeof(f64) * w);
  int *idx = sqlite3_malloc(sizeof(int) * w);
  f64 *xs = sqlite3_malloc(sizeof(f64) * w);
  f64 *sw = sqlite3_malloc(sizeof(f64) * m);
  if (!mu || !sd || !C || !V || !d || !idx || !xs || !sw) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  for (int j = 0; j < w; j++) { /* per-phase (column) mean + std over windows */
    f64 s = 0, s2 = 0;
    for (int i = 0; i < m; i++) {
      f64 v = y[i + j];
      s += v;
      s2 += v * v;
    }
    mu[j] = s / m;
    f64 var = s2 / m - mu[j] * mu[j];
    sd[j] = var > 1e-12 ? sqrt(var) : 1.0;
  }
  for (size_t i = 0; i < (size_t)w * w; i++)
    C[i] = 0;
  for (int i = 0; i < m; i++) { /* covariance of standardized windows */
    for (int j = 0; j < w; j++)
      xs[j] = (y[i + j] - mu[j]) / sd[j];
    for (int a = 0; a < w; a++) {
      f64 xa = xs[a];
      f64 *Ca = C + (size_t)a * w;
      for (int b = a; b < w; b++)
        Ca[b] += xa * xs[b];
    }
  }
  for (int a = 0; a < w; a++) /* symmetrize + normalize */
    for (int b = a; b < w; b++) {
      f64 v = C[(size_t)a * w + b] / (m - 1);
      C[(size_t)a * w + b] = v;
      C[(size_t)b * w + a] = v;
    }
  jacobi_eigen(C, w, d, V);
  for (int j = 0; j < w; j++)
    idx[j] = j;
  for (int i = 1; i < w; i++) { /* insertion sort indices by eigenvalue desc */
    int t = idx[i];
    int jj = i - 1;
    while (jj >= 0 && d[idx[jj]] < d[t]) {
      idx[jj + 1] = idx[jj];
      jj--;
    }
    idx[jj + 1] = t;
  }
  for (int i = 0; i < m; i++) { /* reconstruction error in the top-k subspace */
    f64 norm2 = 0;
    for (int j = 0; j < w; j++) {
      xs[j] = (y[i + j] - mu[j]) / sd[j];
      norm2 += xs[j] * xs[j];
    }
    f64 proj2 = 0;
    for (int t = 0; t < k; t++) {
      int e = idx[t];
      f64 p = 0;
      for (int a = 0; a < w; a++)
        p += xs[a] * V[(size_t)a * w + e];
      proj2 += p * p;
    }
    f64 r = norm2 - proj2;
    sw[i] = r > 0 ? r : 0;
  }
  { /* center-pad the m window scores to n points (matches find_length_rank) */
    int front = w / 2; /* ceil((w-1)/2) */
    for (int t = 0; t < n; t++) {
      int wi = t - front;
      if (wi < 0)
        wi = 0;
      else if (wi >= m)
        wi = m - 1;
      score[t] = sw[wi];
    }
  }

done:
  sqlite3_free(mu);
  sqlite3_free(sd);
  sqlite3_free(C);
  sqlite3_free(V);
  sqlite3_free(d);
  sqlite3_free(idx);
  sqlite3_free(xs);
  sqlite3_free(sw);
  if (rc == SQLITE_NOMEM)
    *errmsg = sqlite3_mprintf("%s: out of memory", PREDICT_ERR_RESOURCE);
  return rc;
}

/* (score, index) pair for turning raw sub-pca scores into percentile ranks.
 * Ties break by index so the ordering is total and the percentiles -- and thus
 * the receipt -- replay exactly regardless of the qsort implementation. */
typedef struct {
  f64 s;
  int i;
} SubpcaRank;
static int subpca_rank_cmp(const void *a, const void *b) {
  const SubpcaRank *x = a, *y = b;
  if (x->s < y->s)
    return -1;
  if (x->s > y->s)
    return 1;
  return x->i < y->i ? -1 : x->i > y->i ? 1 : 0;
}

#pragma endregion

#pragma region detect_anomalies

/* detect_anomalies(query, options) — RFC §4.2.2. Scoring is causal:
 * every point is scored against a one-step-ahead prediction built from
 * PRIOR points only (residual sigma via Welford, updated after each
 * point is scored), so the anomaly cannot hide inside its own fit. */

#define AN_COL_SERIES_KEY 0
#define AN_COL_TS 1
#define AN_COL_VALUE 2
#define AN_COL_FORECAST 3
#define AN_COL_LOWER 4
#define AN_COL_UPPER 5
#define AN_COL_IS 6
#define AN_COL_PROB 7
#define AN_COL_STATUS 8
#define AN_COL_RECEIPT 9
#define AN_COL_QUERY 10
#define AN_COL_OPTIONS 11

typedef struct {
  char *series_key;
  char ts[PREDICT_TS_BUFSIZE];
  f64 value, fc, lo, hi, prob;
  int is_anom;
  int has_pred; /* 0 during warmup */
  int has_values;
  const char *status;
  char receipt_id[PREDICT_ULID_BUFSIZE];
} AnomRow;

typedef struct {
  sqlite3_vtab base;
  sqlite3 *db;
} anom_vtab;

typedef struct {
  sqlite3_vtab_cursor base;
  AnomRow *rows;
  int n_rows;
  int i;
} anom_cursor;

static int anom_row_cmp(const void *a, const void *b) {
  const AnomRow *x = *(AnomRow *const *)a;
  const AnomRow *y = *(AnomRow *const *)b;
  int c = strcmp(x->series_key ? x->series_key : "",
                 y->series_key ? y->series_key : "");
  if (c)
    return c;
  return strcmp(x->ts, y->ts);
}

/* One-step-ahead scorer. Fills pred/sig/valid arrays (index 0 never
 * valid: nothing precedes it). theta_mode swaps the naive base for an
 * SES level over a running seasonal adjustment. */
static void score_online(const f64 *y, int n, int p, int theta_mode,
                         f64 *pred, f64 *sig, u8 *valid) {
  f64 m = 0, M2 = 0; /* Welford over past residuals */
  int c = 0;
  f64 L = y[0]; /* SES level (theta mode) */
  const f64 alpha = 0.3;
  f64 *phase_sum = NULL;
  int *phase_cnt = NULL;
  f64 run_sum = y[0];
  int run_cnt = 1;
  if (theta_mode && p > 1) {
    phase_sum = sqlite3_malloc(sizeof(f64) * p);
    phase_cnt = sqlite3_malloc(sizeof(int) * p);
    if (phase_sum && phase_cnt) {
      memset(phase_sum, 0, sizeof(f64) * p);
      memset(phase_cnt, 0, sizeof(int) * p);
      phase_sum[0] = y[0];
      phase_cnt[0] = 1;
    } else {
      sqlite3_free(phase_sum);
      sqlite3_free(phase_cnt);
      phase_sum = NULL;
      phase_cnt = NULL;
    }
  }

  valid[0] = 0;
  pred[0] = y[0];
  sig[0] = 0;
  for (int t = 1; t < n; t++) {
    f64 pr;
    if (theta_mode) {
      f64 seas = 0;
      if (phase_sum && phase_cnt[t % p] > 0)
        seas = phase_sum[t % p] / phase_cnt[t % p] - run_sum / run_cnt;
      pr = L + seas;
      /* update after prediction */
      f64 deseason = y[t] - seas;
      L += alpha * (deseason - L);
      if (phase_sum) {
        phase_sum[t % p] += y[t];
        phase_cnt[t % p]++;
      }
      run_sum += y[t];
      run_cnt++;
    } else {
      f64 drift = (y[t - 1] - y[0]) / (t > 1 ? (t - 1) : 1);
      if (p > 0 && t >= p)
        pr = y[t - p] + drift * p;
      else
        pr = y[t - 1] + drift;
    }
    pred[t] = pr;
    f64 s = c >= 4 ? sqrt(M2 / c) : 0;
    sig[t] = s;
    valid[t] = (u8)(s > 0 && t >= FORECAST_MIN_HISTORY);

    f64 r = y[t] - pr;
    c++;
    f64 d = r - m;
    m += d / c;
    M2 += d * (r - m);
  }
  sqlite3_free(phase_sum);
  sqlite3_free(phase_cnt);
}

static int an_connect(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  anom_vtab *v = sqlite3_malloc(sizeof(*v));
  if (!v)
    return SQLITE_NOMEM;
  memset(v, 0, sizeof(*v));
  v->db = db;
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(series_key TEXT, ts TEXT, value REAL,"
          " forecast REAL, lower_bound REAL, upper_bound REAL,"
          " is_anomaly INTEGER, anomaly_probability REAL, status TEXT,"
          " receipt_id TEXT, query HIDDEN, options HIDDEN)");
  if (rc != SQLITE_OK) {
    sqlite3_free(v);
    return rc;
  }
  *ppVtab = &v->base;
  return SQLITE_OK;
}

static int an_disconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int an_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pIdx) {
  int seen_query = 0;
  for (int i = 0; i < pIdx->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdx->aConstraint[i];
    if (c->op != SQLITE_INDEX_CONSTRAINT_EQ)
      continue;
    if (c->iColumn == AN_COL_QUERY) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 1;
      pIdx->aConstraintUsage[i].omit = 1;
      seen_query = 1;
    } else if (c->iColumn == AN_COL_OPTIONS) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 2;
      pIdx->aConstraintUsage[i].omit = 1;
    }
  }
  if (!seen_query) {
    pVtab->zErrMsg =
        sqlite3_mprintf("%s: detect_anomalies(query) requires a query",
                        PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  pIdx->estimatedCost = 1000;
  return SQLITE_OK;
}

static int an_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCur) {
  UNUSED_PARAMETER(pVtab);
  anom_cursor *c = sqlite3_malloc(sizeof(*c));
  if (!c)
    return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  *ppCur = &c->base;
  return SQLITE_OK;
}

static void an_rows_free(anom_cursor *c) {
  for (int i = 0; i < c->n_rows; i++)
    sqlite3_free(c->rows[i].series_key);
  sqlite3_free(c->rows);
  c->rows = NULL;
  c->n_rows = 0;
  c->i = 0;
}

static int an_close(sqlite3_vtab_cursor *pCur) {
  anom_cursor *c = (anom_cursor *)pCur;
  an_rows_free(c);
  sqlite3_free(c);
  return SQLITE_OK;
}

static int an_error(anom_cursor *cur, const char *code, const char *msg,
                    const char *detail) {
  sqlite3_free(cur->base.pVtab->zErrMsg);
  cur->base.pVtab->zErrMsg =
      sqlite3_mprintf("%s: %s%s", code, msg, detail ? detail : "");
  return SQLITE_ERROR;
}

/* Resolve the detect_anomalies model choice. Sets *use_subpca,
 * *theta_mode, and *model_id (static string). On failure sets *errmsg
 * ("CODE: detail"). Shared by the query and aggregate forms. */
static int an_resolve_model(const ForecastOpts *opts, int *use_subpca,
                            int *theta_mode, const char **model_id,
                            char **errmsg) {
  /* sub-pca is a subsequence detector, not a forecast model: it has its own
   * scoring path and bypasses the one-step-ahead forecaster resolution. */
  *use_subpca = opts->model && strcmp(opts->model, "sub-pca") == 0;
  *theta_mode = 1; /* default-ts and theta-classic */
  *model_id = *use_subpca ? "sub-pca" : resolve_ts_model_id(opts->model);
  if (!*model_id) {
    *errmsg = sqlite3_mprintf("%s: no such model: %s",
                              PREDICT_ERR_MODEL_NOT_FOUND, opts->model);
    return SQLITE_ERROR;
  }
  /* the anomaly scorer implements only the one-step forecasters (theta /
   * seasonal-naive) and sub-pca; a forecast-only model such as tsb must be
   * rejected here rather than silently scored with theta */
  if (!*use_subpca && strcmp(*model_id, "theta-classic") != 0 &&
      strcmp(*model_id, "stub-seasonal-naive") != 0) {
    *errmsg = sqlite3_mprintf(
        "%s: model not supported for detect_anomalies (use theta-classic, "
        "stub-seasonal-naive, or sub-pca): %s",
        PREDICT_ERR_OPTIONS, opts->model);
    return SQLITE_ERROR;
  }
  if (!*use_subpca && strcmp(*model_id, "stub-seasonal-naive") == 0)
    *theta_mode = 0;
  return SQLITE_OK;
}

/* Score every collected series into AnomRow output rows
 * (sqlite3_malloc'd into *rows_out). Returns SQLITE_OK or SQLITE_NOMEM;
 * on failure partial rows are freed here and *rows_out is NULL. */
static int an_run_series(SeriesBuf *series, int n_series,
                         const ForecastOpts *opts, int use_subpca,
                         int theta_mode, AnomRow **rows_out, int *n_rows_out) {
  *rows_out = NULL;
  *n_rows_out = 0;
  int rc = SQLITE_OK;

  f64 z_thr = predict0_norm_quantile(0.5 + opts->confidence / 2);
  int max_rows = 0;
  for (int i = 0; i < n_series; i++) {
    SeriesBuf *s = &series[i];
    int usable = !s->non_numeric && s->n >= FORECAST_MIN_HISTORY;
    int scored = (opts->context_limit > 0 && s->n > opts->context_limit)
                     ? opts->context_limit
                     : s->n;
    max_rows += usable ? scored : 1;
  }
  AnomRow *rows = sqlite3_malloc(sizeof(AnomRow) * (max_rows ? max_rows : 1));
  int n_rows = 0;
  if (!rows)
    return SQLITE_NOMEM;

  for (int i = 0; i < n_series; i++) {
    SeriesBuf *s = &series[i];
    const char *status = NULL;
    if (s->non_numeric)
      status = "non_numeric";
    else if (s->n < FORECAST_MIN_HISTORY)
      status = "insufficient_history";
    if (status) {
      AnomRow *r = &rows[n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      r->status = status;
      continue;
    }
    series_sort(s);
    int truncated = 0;
    f64 *y = s->val;
    i64 *ts = s->ts;
    int n = s->n;
    if (opts->context_limit > 0 && n > opts->context_limit) {
      y += n - opts->context_limit;
      ts += n - opts->context_limit;
      n = opts->context_limit;
      truncated = 1;
    }
    if (use_subpca) { /* subsequence reconstruction detector (not a forecaster) */
      f64 *score = sqlite3_malloc(sizeof(f64) * n);
      SubpcaRank *rk = sqlite3_malloc(sizeof(SubpcaRank) * n);
      if (!score || !rk) {
        sqlite3_free(score);
        sqlite3_free(rk);
        rc = SQLITE_NOMEM;
        break;
      }
      int win = 0;
      char *serr = NULL;
      if (subpca_score(y, n, &win, score, &serr) != SQLITE_OK) {
        sqlite3_free(serr);
        sqlite3_free(score);
        sqlite3_free(rk);
        AnomRow *r = &rows[n_rows++];
        memset(r, 0, sizeof(*r));
        r->series_key = sqlite3_mprintf("%s", s->key);
        r->status = "insufficient_history";
        continue;
      }
      for (int t = 0; t < n; t++) {
        rk[t].s = score[t];
        rk[t].i = t;
      }
      qsort(rk, n, sizeof(SubpcaRank), subpca_rank_cmp);
      /* anomaly_probability = percentile rank: "more anomalous than this
       * fraction of the series". Rank-preserving, so is_anomaly = rank >
       * threshold flags the top (1 - threshold) fraction. */
      for (int r0 = 0; r0 < n; r0++)
        score[rk[r0].i] = n > 1 ? (f64)r0 / (n - 1) : 0.0;
      for (int t = 0; t < n; t++) {
        AnomRow *r = &rows[n_rows++];
        memset(r, 0, sizeof(*r));
        r->series_key = sqlite3_mprintf("%s", s->key);
        predict0_format_timestamp(ts[t], r->ts, sizeof(r->ts));
        r->value = y[t];
        r->has_values = 1; /* has_pred stays 0: no forecast/interval */
        r->status = truncated ? "truncated" : "ok";
        r->prob = score[t];
        r->is_anom = r->prob > opts->confidence;
      }
      sqlite3_free(score);
      sqlite3_free(rk);
      continue;
    }
    int p = detect_period(y, n);
    f64 *pred = sqlite3_malloc(sizeof(f64) * n);
    f64 *sig = sqlite3_malloc(sizeof(f64) * n);
    u8 *valid = sqlite3_malloc(n);
    if (!pred || !sig || !valid) {
      sqlite3_free(pred);
      sqlite3_free(sig);
      sqlite3_free(valid);
      rc = SQLITE_NOMEM;
      break;
    }
    score_online(y, n, p, theta_mode, pred, sig, valid);
    for (int t = 0; t < n; t++) {
      AnomRow *r = &rows[n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      predict0_format_timestamp(ts[t], r->ts, sizeof(r->ts));
      r->value = y[t];
      r->has_values = 1;
      r->status = truncated ? "truncated" : "ok";
      if (valid[t]) {
        r->has_pred = 1;
        r->fc = pred[t];
        r->lo = pred[t] - z_thr * sig[t];
        r->hi = pred[t] + z_thr * sig[t];
        f64 z = (y[t] - pred[t]) / sig[t];
        r->prob = erf(fabs(z) / 1.4142135623730951);
        r->is_anom = r->prob > opts->confidence;
      }
    }
    sqlite3_free(pred);
    sqlite3_free(sig);
    sqlite3_free(valid);
  }

  if (rc != SQLITE_OK) {
    for (int i = 0; i < n_rows; i++)
      sqlite3_free(rows[i].series_key);
    sqlite3_free(rows);
    return rc;
  }
  *rows_out = rows;
  *n_rows_out = n_rows;
  return SQLITE_OK;
}

/* RFC §4.1.3 result hash over rows sorted by (series_key, ts).
 * Returns SQLITE_OK or SQLITE_NOMEM. */
static int an_result_hash(AnomRow *rows, int n_rows,
                          char out[PREDICT_HEX_BUFSIZE]) {
  AnomRow **order = sqlite3_malloc(sizeof(AnomRow *) * (n_rows ? n_rows : 1));
  if (!order)
    return SQLITE_NOMEM;
  for (int i = 0; i < n_rows; i++)
    order[i] = &rows[i];
  qsort(order, (usize)n_rows, sizeof(AnomRow *), anom_row_cmp);
  predict0_hasher h;
  predict0_hash_init(&h);
  for (int i = 0; i < n_rows; i++) {
    AnomRow *r = order[i];
    predict0_hash_text(&h, r->series_key);
    if (r->has_values) {
      predict0_hash_text(&h, r->ts);
      predict0_hash_real(&h, r->value);
    } else {
      predict0_hash_null(&h);
      predict0_hash_null(&h);
    }
    if (r->has_pred) {
      predict0_hash_real(&h, r->fc);
      predict0_hash_real(&h, r->lo);
      predict0_hash_real(&h, r->hi);
    } else {
      predict0_hash_null(&h);
      predict0_hash_null(&h);
      predict0_hash_null(&h);
    }
    if (r->has_values) {
      predict0_hash_int(&h, r->is_anom);
      predict0_hash_real(&h, r->prob);
    } else {
      predict0_hash_null(&h);
      predict0_hash_null(&h);
    }
    predict0_hash_row_end(&h);
  }
  sqlite3_free(order);
  predict0_hash_hex(&h, out);
  return SQLITE_OK;
}

/* Canonical detect_anomalies receipt params. agg_form omits the
 * query-shape keys (RFC §4.2.8). Returns sqlite3_malloc'd JSON or NULL. */
static char *an_build_params(sqlite3 *db, const ForecastOpts *opts,
                             const char *model_id, const char *time_col,
                             const char *value_col, int agg_form) {
  char group_json[600] = "";
  if (opts->n_group_cols) {
    usize gl = 0;
    gl += (usize)snprintf(group_json + gl, sizeof(group_json) - gl, "[");
    for (int g = 0; g < opts->n_group_cols && gl < sizeof(group_json) - 4; g++)
      gl += (usize)snprintf(group_json + gl, sizeof(group_json) - gl,
                            "%s\"%s\"", g ? "," : "", opts->group_cols[g]);
    snprintf(group_json + gl, sizeof(group_json) - gl, "]");
  }
  sqlite3_stmt *pj = NULL;
  char *params = NULL;
  const char *sql =
      agg_form ? "SELECT json_object('anomaly_prob_threshold', ?1,"
                 " 'context_limit', ?2, 'model', ?4, 'receipt', 1)"
               : "SELECT json_object('anomaly_prob_threshold', ?1,"
                 " 'context_limit', ?2,"
                 " 'group_cols', CASE WHEN ?3 = '' THEN NULL ELSE json(?3) END,"
                 " 'model', ?4, 'receipt', 1, 'time_col', ?5, 'value_col', ?6)";
  if (sqlite3_prepare_v2(db, sql, -1, &pj, NULL) == SQLITE_OK) {
    sqlite3_bind_double(pj, 1, opts->confidence);
    /* 0 = "score every point": record as null so replay re-defaults rather
     * than failing the context_limit >= 1 check on the recorded params */
    if (opts->context_limit > 0)
      sqlite3_bind_int(pj, 2, opts->context_limit);
    else
      sqlite3_bind_null(pj, 2);
    if (!agg_form)
      sqlite3_bind_text(pj, 3, group_json, -1, SQLITE_STATIC);
    sqlite3_bind_text(pj, 4, model_id, -1, SQLITE_STATIC);
    if (!agg_form) {
      if (time_col)
        sqlite3_bind_text(pj, 5, time_col, -1, SQLITE_STATIC);
      if (value_col)
        sqlite3_bind_text(pj, 6, value_col, -1, SQLITE_STATIC);
    }
    if (sqlite3_step(pj) == SQLITE_ROW)
      params = sqlite3_mprintf("%s", (const char *)sqlite3_column_text(pj, 0));
    sqlite3_finalize(pj);
  }
  return params;
}

static int an_filter(sqlite3_vtab_cursor *pCur, int idxNum,
                     const char *idxStr, int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  anom_cursor *cur = (anom_cursor *)pCur;
  anom_vtab *vtab = (anom_vtab *)cur->base.pVtab;
  sqlite3 *db = vtab->db;
  an_rows_free(cur);

  if (argc < 1)
    return an_error(cur, PREDICT_ERR_SCHEMA, "query required", NULL);
  const char *query = (const char *)sqlite3_value_text(argv[0]);
  if (!query)
    return an_error(cur, PREDICT_ERR_SCHEMA, "query must be text", NULL);

  ForecastOpts opts;
  memset(&opts, 0, sizeof(opts));
  opts.confidence = 0.99; /* anomaly_prob_threshold default */
  /* 0 = score every point. Unlike forecasting (which only needs recent
   * context), anomaly detection must scan the whole series, so it does NOT
   * default to FORECAST_DEFAULT_CONTEXT_LIMIT -- that would silently drop all
   * but the last N points. An explicit context_limit still caps it. */
  opts.context_limit = 0;
  opts.receipt = 1;

  const char *options_json =
      argc >= 2 ? (const char *)sqlite3_value_text(argv[1]) : NULL;
  char *errmsg = NULL;
  if (predict0_options_parse(db, options_json, ANOMALY_OPTION_KEYS,
                             forecast_opt_cb, &opts, &errmsg)) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = errmsg;
    forecast_opts_free(&opts);
    return SQLITE_ERROR;
  }
  int use_subpca = 0, theta_mode = 1;
  const char *model_id = NULL;
  char *mrerr = NULL;
  if (an_resolve_model(&opts, &use_subpca, &theta_mode, &model_id, &mrerr) !=
      SQLITE_OK) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = mrerr;
    forecast_opts_free(&opts);
    return SQLITE_ERROR;
  }

  /* prepare + collect the series (shared with forecast) */
  SeriesBuf *series = NULL;
  int n_series = 0;
  char *resolved_time = NULL, *resolved_value = NULL;
  char *cerr = NULL;
  int rc = collect_series(db, query, &opts, &series, &n_series,
                          &resolved_time, &resolved_value, &cerr);

#define AN_FREE_SERIES()                                                      \
  do {                                                                        \
    series_array_free(series, n_series);                                      \
    sqlite3_free(resolved_time);                                             \
    sqlite3_free(resolved_value);                                            \
    forecast_opts_free(&opts);                                               \
  } while (0)

  if (rc != SQLITE_OK) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = cerr;
    forecast_opts_free(&opts);
    return rc;
  }

  /* score every series */
  rc = an_run_series(series, n_series, &opts, use_subpca, theta_mode,
                     &cur->rows, &cur->n_rows);
  if (rc != SQLITE_OK) {
    AN_FREE_SERIES();
    return rc;
  }

  /* receipt */
  if (opts.receipt) {
    char *rerr = NULL;
    char result_hash[PREDICT_HEX_BUFSIZE];
    if (an_result_hash(cur->rows, cur->n_rows, result_hash) != SQLITE_OK) {
      AN_FREE_SERIES();
      an_rows_free(cur);
      return SQLITE_NOMEM;
    }
    char *params = an_build_params(db, &opts, model_id, resolved_time,
                                   resolved_value, 0);
    if (!params) {
      AN_FREE_SERIES();
      an_rows_free(cur);
      return an_error(cur, PREDICT_ERR_RESOURCE,
                      "could not canonicalize params", NULL);
    }
    char receipt_id[PREDICT_ULID_BUFSIZE];
    int irc = predict0_emit_receipt(db, "detect_anomalies", model_id, params,
                                    query, result_hash, receipt_id, &rerr);
    sqlite3_free(params);
    if (irc != SQLITE_OK) {
      sqlite3_free(vtab->base.zErrMsg);
      vtab->base.zErrMsg = rerr;
      AN_FREE_SERIES();
      an_rows_free(cur);
      return SQLITE_ERROR;
    }
    for (int i = 0; i < cur->n_rows; i++)
      memcpy(cur->rows[i].receipt_id, receipt_id, sizeof(receipt_id));
  }

  AN_FREE_SERIES();
#undef AN_FREE_SERIES
  cur->i = 0;
  return SQLITE_OK;
}

static int an_next(sqlite3_vtab_cursor *pCur) {
  ((anom_cursor *)pCur)->i++;
  return SQLITE_OK;
}

static int an_eof(sqlite3_vtab_cursor *pCur) {
  anom_cursor *c = (anom_cursor *)pCur;
  return c->i >= c->n_rows;
}

static int an_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx,
                     int col) {
  anom_cursor *c = (anom_cursor *)pCur;
  AnomRow *r = &c->rows[c->i];
  switch (col) {
  case AN_COL_SERIES_KEY:
    sqlite3_result_text(ctx, r->series_key ? r->series_key : "", -1,
                        SQLITE_TRANSIENT);
    break;
  case AN_COL_TS:
    if (r->has_values)
      sqlite3_result_text(ctx, r->ts, -1, SQLITE_TRANSIENT);
    break;
  case AN_COL_VALUE:
    if (r->has_values)
      sqlite3_result_double(ctx, r->value);
    break;
  case AN_COL_FORECAST:
    if (r->has_pred)
      sqlite3_result_double(ctx, r->fc);
    break;
  case AN_COL_LOWER:
    if (r->has_pred)
      sqlite3_result_double(ctx, r->lo);
    break;
  case AN_COL_UPPER:
    if (r->has_pred)
      sqlite3_result_double(ctx, r->hi);
    break;
  case AN_COL_IS:
    if (r->has_values)
      sqlite3_result_int(ctx, r->is_anom);
    break;
  case AN_COL_PROB:
    if (r->has_values)
      sqlite3_result_double(ctx, r->prob);
    break;
  case AN_COL_STATUS:
    sqlite3_result_text(ctx, r->status, -1, SQLITE_STATIC);
    break;
  case AN_COL_RECEIPT:
    if (r->receipt_id[0])
      sqlite3_result_text(ctx, r->receipt_id, -1, SQLITE_TRANSIENT);
    break;
  default:
    break;
  }
  return SQLITE_OK;
}

static int an_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  *pRowid = ((anom_cursor *)pCur)->i;
  return SQLITE_OK;
}

static sqlite3_module anomModule = {
    /* iVersion    */ 0,
    /* xCreate     */ NULL,
    /* xConnect    */ an_connect,
    /* xBestIndex  */ an_best_index,
    /* xDisconnect */ an_disconnect,
    /* xDestroy    */ NULL,
    /* xOpen       */ an_open,
    /* xClose      */ an_close,
    /* xFilter     */ an_filter,
    /* xNext       */ an_next,
    /* xEof        */ an_eof,
    /* xColumn     */ an_column,
    /* xRowid      */ an_rowid,
    /* xUpdate     */ NULL,
    /* xBegin      */ NULL,
    /* xSync       */ NULL,
    /* xCommit     */ NULL,
    /* xRollback   */ NULL,
    /* xFindMethod */ NULL,
    /* xRename     */ NULL,
    /* xSavepoint  */ NULL,
    /* xRelease    */ NULL,
    /* xRollbackTo */ NULL,
    /* xShadowName */ NULL,
    /* xIntegrity  */ NULL};

#pragma endregion

#pragma region backtest

/* backtest(query, horizon, options) — eponymous table-valued function. Rolling-
 * origin evaluation of a statistical model (or auto) over each series: for each
 * fold it reports point-accuracy metrics and, for the chosen interval_method,
 * interval coverage + width. This is the on-device audit primitive: an agent
 * scores its own forecasts, and validates conformal coverage, locally. */

#define BT_COL_SERIES_KEY 0
#define BT_COL_FOLD 1
#define BT_COL_CUTOFF 2
#define BT_COL_MODEL 3
#define BT_COL_N 4
#define BT_COL_MAE 5
#define BT_COL_RMSE 6
#define BT_COL_MASE 7
#define BT_COL_SMAPE 8
#define BT_COL_COVERAGE 9
#define BT_COL_WIDTH 10
#define BT_COL_STATUS 11
#define BT_COL_RECEIPT 12
#define BT_COL_QUERY 13
#define BT_COL_HORIZON 14
#define BT_COL_OPTIONS 15

typedef struct {
  char *series_key;
  int fold;
  char cutoff[PREDICT_TS_BUFSIZE];
  const char *model; /* static id */
  int n;
  f64 mae, rmse, mase, smape, coverage, width;
  int has_metrics; /* 0 for status-only rows */
  int has_band;    /* 0 when no interval was computed */
  const char *status;
  char receipt_id[PREDICT_ULID_BUFSIZE];
} BtRow;

static int bt_row_cmp(const void *a, const void *b) {
  const BtRow *x = *(BtRow *const *)a;
  const BtRow *y = *(BtRow *const *)b;
  int c = strcmp(x->series_key ? x->series_key : "",
                 y->series_key ? y->series_key : "");
  if (c)
    return c;
  return x->fold - y->fold;
}

typedef struct {
  sqlite3_vtab base;
  sqlite3 *db;
} bt_vtab;

typedef struct {
  sqlite3_vtab_cursor base;
  BtRow *rows;
  int n_rows;
  int i;
} bt_cursor;

static int bt_connect(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                      sqlite3_vtab **ppVtab, char **pzErr) {
  UNUSED_PARAMETER(pAux);
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  bt_vtab *v = sqlite3_malloc(sizeof(*v));
  if (!v)
    return SQLITE_NOMEM;
  memset(v, 0, sizeof(*v));
  v->db = db;
  int rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(series_key TEXT, fold INTEGER, cutoff_timestamp TEXT,"
          " model TEXT, n INTEGER, mae REAL, rmse REAL, mase REAL, smape REAL,"
          " coverage REAL, mean_interval_width REAL, status TEXT,"
          " receipt_id TEXT, query HIDDEN, horizon HIDDEN, options HIDDEN)");
  if (rc != SQLITE_OK) {
    sqlite3_free(v);
    return rc;
  }
  *ppVtab = &v->base;
  return SQLITE_OK;
}

static int bt_disconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int bt_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pIdx) {
  int seen_query = 0, seen_horizon = 0;
  for (int i = 0; i < pIdx->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdx->aConstraint[i];
    if (c->op != SQLITE_INDEX_CONSTRAINT_EQ)
      continue;
    if (c->iColumn == BT_COL_QUERY) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 1;
      pIdx->aConstraintUsage[i].omit = 1;
      seen_query = 1;
    } else if (c->iColumn == BT_COL_HORIZON) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 2;
      pIdx->aConstraintUsage[i].omit = 1;
      seen_horizon = 1;
    } else if (c->iColumn == BT_COL_OPTIONS) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 3;
      pIdx->aConstraintUsage[i].omit = 1;
    }
  }
  if (!seen_query || !seen_horizon) {
    pVtab->zErrMsg = sqlite3_mprintf(
        "%s: backtest(query, horizon) requires both arguments",
        PREDICT_ERR_SCHEMA);
    return SQLITE_ERROR;
  }
  pIdx->estimatedCost = 1000;
  return SQLITE_OK;
}

static int bt_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCur) {
  UNUSED_PARAMETER(pVtab);
  bt_cursor *c = sqlite3_malloc(sizeof(*c));
  if (!c)
    return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  *ppCur = &c->base;
  return SQLITE_OK;
}

static void bt_rows_free(bt_cursor *c) {
  for (int i = 0; i < c->n_rows; i++)
    sqlite3_free(c->rows[i].series_key);
  sqlite3_free(c->rows);
  c->rows = NULL;
  c->n_rows = 0;
  c->i = 0;
}

static int bt_close(sqlite3_vtab_cursor *pCur) {
  bt_cursor *c = (bt_cursor *)pCur;
  bt_rows_free(c);
  sqlite3_free(c);
  return SQLITE_OK;
}

static int bt_error(bt_cursor *cur, const char *code, const char *msg,
                    const char *detail) {
  sqlite3_free(cur->base.pVtab->zErrMsg);
  cur->base.pVtab->zErrMsg =
      sqlite3_mprintf("%s: %s%s", code, msg, detail ? detail : "");
  return SQLITE_ERROR;
}

/* Conformal leave-fold-out absolute-residual quantile for step h, excluding
 * fold `skip`. col is scratch [nf]. Returns the quantile at level conf. */
static f64 bt_lofo_q(const f64 *act, const f64 *fcst, int nf, int horizon, int h,
                     int skip, f64 conf, f64 *col) {
  int m = 0;
  for (int g = 0; g < nf; g++)
    if (g != skip)
      col[m++] = fabs(act[g * horizon + h] - fcst[g * horizon + h]);
  for (int i = 1; i < m; i++) { /* insertion sort (m small) */
    f64 v = col[i];
    int j = i - 1;
    while (j >= 0 && col[j] > v) {
      col[j + 1] = col[j];
      j--;
    }
    col[j + 1] = v;
  }
  int rank = (int)ceil((f64)(m + 1) * conf);
  if (rank > m)
    rank = m;
  if (rank < 1)
    rank = 1;
  return col[rank - 1];
}

static int bt_filter(sqlite3_vtab_cursor *pCur, int idxNum, const char *idxStr,
                     int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  bt_cursor *cur = (bt_cursor *)pCur;
  bt_vtab *vtab = (bt_vtab *)cur->base.pVtab;
  sqlite3 *db = vtab->db;
  bt_rows_free(cur);

  if (argc < 2)
    return bt_error(cur, PREDICT_ERR_SCHEMA,
                    "backtest(query, horizon) requires both arguments", NULL);
  const char *query = (const char *)sqlite3_value_text(argv[0]);
  if (!query)
    return bt_error(cur, PREDICT_ERR_SCHEMA, "query must be text", NULL);
  if (sqlite3_value_type(argv[1]) != SQLITE_INTEGER)
    return bt_error(cur, PREDICT_ERR_HORIZON, "horizon must be an integer", NULL);
  int horizon = sqlite3_value_int(argv[1]);
  if (horizon < 1 || horizon > FORECAST_MAX_HORIZON)
    return bt_error(cur, PREDICT_ERR_HORIZON, "horizon out of range 1..1000",
                    NULL);

  ForecastOpts opts;
  memset(&opts, 0, sizeof(opts));
  opts.confidence = 0.95;
  opts.context_limit = FORECAST_DEFAULT_CONTEXT_LIMIT;
  opts.receipt = 1;

  const char *options_json =
      argc >= 3 ? (const char *)sqlite3_value_text(argv[2]) : NULL;
  char *errmsg = NULL;
  if (predict0_options_parse(db, options_json, BACKTEST_OPTION_KEYS,
                             forecast_opt_cb, &opts, &errmsg)) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = errmsg;
    forecast_opts_free(&opts);
    return SQLITE_ERROR;
  }

  int is_auto = opts.model && strcmp(opts.model, "auto") == 0;
  predict0_ts_model fixed_model = is_auto ? NULL : resolve_ts_model(opts.model);
  const char *fixed_id = is_auto ? "auto" : resolve_ts_model_id(opts.model);
  if (!is_auto && !fixed_model) {
    int rc = bt_error(cur, PREDICT_ERR_MODEL_NOT_FOUND,
                      "backtest() supports the statistical models "
                      "(theta-classic, stub-seasonal-naive) or auto: ",
                      opts.model);
    forecast_opts_free(&opts);
    return rc;
  }

  SeriesBuf *series = NULL;
  int n_series = 0;
  char *resolved_time = NULL, *resolved_value = NULL, *cerr = NULL;
  int rc = collect_series(db, query, &opts, &series, &n_series, &resolved_time,
                          &resolved_value, &cerr);

#define BT_FREE_SERIES()                                                       \
  do {                                                                         \
    series_array_free(series, n_series);                                       \
    sqlite3_free(resolved_time);                                               \
    sqlite3_free(resolved_value);                                              \
    forecast_opts_free(&opts);                                                 \
  } while (0)

  if (rc != SQLITE_OK) {
    sqlite3_free(vtab->base.zErrMsg);
    vtab->base.zErrMsg = cerr;
    forecast_opts_free(&opts);
    return rc;
  }

  int folds_req = opts.folds > 0 ? opts.folds : FORECAST_DEFAULT_FOLDS;
  f64 z = predict0_norm_quantile(0.5 + opts.confidence / 2);

  int max_rows = 0;
  for (int i = 0; i < n_series; i++) {
    SeriesBuf *s = &series[i];
    int usable = !s->non_numeric && s->n >= FORECAST_MIN_HISTORY;
    max_rows += usable ? folds_req : 1;
  }
  cur->rows = sqlite3_malloc(sizeof(BtRow) * (max_rows ? max_rows : 1));
  f64 *act = sqlite3_malloc(sizeof(f64) * (size_t)folds_req * horizon);
  f64 *fcst = sqlite3_malloc(sizeof(f64) * (size_t)folds_req * horizon);
  f64 *sig = sqlite3_malloc(sizeof(f64) * (size_t)folds_req * horizon);
  f64 *col = sqlite3_malloc(sizeof(f64) * folds_req);
  i64 *cutoff_ms = sqlite3_malloc(sizeof(i64) * folds_req);
  if (!cur->rows || !act || !fcst || !sig || !col || !cutoff_ms)
    rc = SQLITE_NOMEM;
  cur->n_rows = 0;

  for (int i = 0; rc == SQLITE_OK && i < n_series; i++) {
    SeriesBuf *s = &series[i];
    const char *status = NULL;
    if (s->non_numeric)
      status = "non_numeric";
    else if (s->n < FORECAST_MIN_HISTORY)
      status = "insufficient_history";
    if (status) {
      BtRow *r = &cur->rows[cur->n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      r->status = status;
      continue;
    }
    series_sort(s);
    int truncated = 0;
    f64 *y = s->val;
    i64 *ts = s->ts;
    int n = s->n;
    if (opts.context_limit > 0 && n > opts.context_limit) {
      y += n - opts.context_limit;
      ts += n - opts.context_limit;
      n = opts.context_limit;
      truncated = 1;
    }

    predict0_ts_model use_model = fixed_model;
    const char *use_id = fixed_id;
    f64 scale = naive_scale(y, n);
    if (is_auto) {
      int bi = ts_auto_select(y, ts, n, horizon, opts.gap, folds_req, scale, act,
                              fcst);
      if (bi <= -2) {
        rc = SQLITE_NOMEM;
        break;
      }
      use_model = bi >= 0 ? TS_MODELS[bi].run : model_theta;
      use_id = bi >= 0 ? TS_MODELS[bi].id : "theta-classic";
    }
    int nf = ts_backtest(use_model, y, ts, n, horizon, opts.gap, folds_req, act,
                         fcst, sig, cutoff_ms);
    if (nf <= -2) {
      rc = SQLITE_NOMEM;
      break;
    }
    if (nf == 0) { /* too short to form a single fold */
      BtRow *r = &cur->rows[cur->n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      r->status = "insufficient_history";
      continue;
    }
    for (int f = 0; f < nf; f++) {
      const f64 *fa = act + (size_t)f * horizon;
      const f64 *ff = fcst + (size_t)f * horizon;
      const f64 *fs = sig + (size_t)f * horizon;
      BtRow *r = &cur->rows[cur->n_rows++];
      memset(r, 0, sizeof(*r));
      r->series_key = sqlite3_mprintf("%s", s->key);
      r->fold = f;
      predict0_format_timestamp(cutoff_ms[f], r->cutoff, sizeof(r->cutoff));
      r->model = use_id;
      r->n = horizon;
      r->mae = metric_mae(fa, ff, horizon);
      r->rmse = metric_rmse(fa, ff, horizon);
      r->mase = scale > 0 ? r->mae / scale : 0;
      r->smape = metric_smape(fa, ff, horizon);
      r->has_metrics = 1;
      r->status = truncated ? "truncated" : "ok";
      f64 cov = 0, wid = 0;
      if (!opts.interval_conformal) {
        for (int h = 0; h < horizon; h++) {
          f64 lo = ff[h] - z * fs[h], hi = ff[h] + z * fs[h];
          wid += hi - lo;
          if (fa[h] >= lo && fa[h] <= hi)
            cov += 1;
        }
        r->coverage = cov / horizon;
        r->width = wid / horizon;
        r->has_band = 1;
      } else if (nf >= CONFORMAL_MIN_FOLDS) { /* leave-fold-out conformal band */
        for (int h = 0; h < horizon; h++) {
          f64 q = bt_lofo_q(act, fcst, nf, horizon, h, f, opts.confidence, col);
          f64 lo = ff[h] - q, hi = ff[h] + q;
          wid += hi - lo;
          if (fa[h] >= lo && fa[h] <= hi)
            cov += 1;
        }
        r->coverage = cov / horizon;
        r->width = wid / horizon;
        r->has_band = 1;
      }
    }
  }

  sqlite3_free(act);
  sqlite3_free(fcst);
  sqlite3_free(sig);
  sqlite3_free(col);
  sqlite3_free(cutoff_ms);

  if (rc != SQLITE_OK) {
    BT_FREE_SERIES();
    bt_rows_free(cur);
    return rc;
  }

  if (opts.receipt) {
    char *rerr = NULL;
    BtRow **order =
        sqlite3_malloc(sizeof(BtRow *) * (cur->n_rows ? cur->n_rows : 1));
    if (!order) {
      BT_FREE_SERIES();
      bt_rows_free(cur);
      return SQLITE_NOMEM;
    }
    for (int i = 0; i < cur->n_rows; i++)
      order[i] = &cur->rows[i];
    qsort(order, (usize)cur->n_rows, sizeof(BtRow *), bt_row_cmp);
    predict0_hasher h;
    predict0_hash_init(&h);
    for (int i = 0; i < cur->n_rows; i++) {
      BtRow *r = order[i];
      predict0_hash_text(&h, r->series_key);
      if (r->has_metrics) {
        predict0_hash_int(&h, r->fold);
        predict0_hash_text(&h, r->cutoff);
        predict0_hash_text(&h, r->model);
        predict0_hash_int(&h, r->n);
        predict0_hash_real(&h, r->mae);
        predict0_hash_real(&h, r->rmse);
        predict0_hash_real(&h, r->mase);
        predict0_hash_real(&h, r->smape);
        if (r->has_band) {
          predict0_hash_real(&h, r->coverage);
          predict0_hash_real(&h, r->width);
        } else {
          predict0_hash_null(&h);
          predict0_hash_null(&h);
        }
      } else {
        for (int k = 0; k < 10; k++)
          predict0_hash_null(&h);
      }
      predict0_hash_row_end(&h);
    }
    sqlite3_free(order);
    char result_hash[PREDICT_HEX_BUFSIZE];
    predict0_hash_hex(&h, result_hash);

    char group_json[600] = "";
    if (opts.n_group_cols) {
      usize gl = 0;
      gl += (usize)snprintf(group_json + gl, sizeof(group_json) - gl, "[");
      for (int g = 0; g < opts.n_group_cols && gl < sizeof(group_json) - 4; g++)
        gl += (usize)snprintf(group_json + gl, sizeof(group_json) - gl,
                              "%s\"%s\"", g ? "," : "", opts.group_cols[g]);
      snprintf(group_json + gl, sizeof(group_json) - gl, "]");
    }
    sqlite3_stmt *pj = NULL;
    char *params = NULL;
    if (sqlite3_prepare_v2(
            db,
            "SELECT json_object('confidence_level', ?1, 'context_limit', ?2,"
            " 'folds', ?3, 'gap', ?4,"
            " 'group_cols', CASE WHEN ?5 = '' THEN NULL ELSE json(?5) END,"
            " 'horizon', ?6, 'interval_method', ?7, 'model', ?8,"
            " 'receipt', 1, 'time_col', ?9, 'value_col', ?10)",
            -1, &pj, NULL) == SQLITE_OK) {
      sqlite3_bind_double(pj, 1, opts.confidence);
      sqlite3_bind_int(pj, 2, opts.context_limit);
      if (opts.folds > 0)
        sqlite3_bind_int(pj, 3, opts.folds);
      else
        sqlite3_bind_null(pj, 3);
      if (opts.gap > 0)
        sqlite3_bind_int(pj, 4, opts.gap);
      else
        sqlite3_bind_null(pj, 4);
      sqlite3_bind_text(pj, 5, group_json, -1, SQLITE_STATIC);
      sqlite3_bind_int(pj, 6, horizon);
      sqlite3_bind_text(pj, 7, opts.interval_conformal ? "conformal" : "residual",
                        -1, SQLITE_STATIC);
      sqlite3_bind_text(pj, 8, fixed_id, -1, SQLITE_STATIC);
      if (resolved_time)
        sqlite3_bind_text(pj, 9, resolved_time, -1, SQLITE_STATIC);
      if (resolved_value)
        sqlite3_bind_text(pj, 10, resolved_value, -1, SQLITE_STATIC);
      if (sqlite3_step(pj) == SQLITE_ROW)
        params = sqlite3_mprintf("%s", (const char *)sqlite3_column_text(pj, 0));
      sqlite3_finalize(pj);
    }
    if (!params) {
      BT_FREE_SERIES();
      bt_rows_free(cur);
      return bt_error(cur, PREDICT_ERR_RESOURCE, "could not canonicalize params",
                      NULL);
    }
    char receipt_id[PREDICT_ULID_BUFSIZE];
    int irc = predict0_emit_receipt(db, "backtest", fixed_id, params, query,
                                    result_hash, receipt_id, &rerr);
    sqlite3_free(params);
    if (irc != SQLITE_OK) {
      sqlite3_free(vtab->base.zErrMsg);
      vtab->base.zErrMsg = rerr;
      BT_FREE_SERIES();
      bt_rows_free(cur);
      return SQLITE_ERROR;
    }
    for (int i = 0; i < cur->n_rows; i++)
      memcpy(cur->rows[i].receipt_id, receipt_id, sizeof(receipt_id));
  }

  BT_FREE_SERIES();
#undef BT_FREE_SERIES
  cur->i = 0;
  return SQLITE_OK;
}

static int bt_next(sqlite3_vtab_cursor *pCur) {
  ((bt_cursor *)pCur)->i++;
  return SQLITE_OK;
}

static int bt_eof(sqlite3_vtab_cursor *pCur) {
  bt_cursor *c = (bt_cursor *)pCur;
  return c->i >= c->n_rows;
}

static int bt_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx, int col) {
  bt_cursor *c = (bt_cursor *)pCur;
  BtRow *r = &c->rows[c->i];
  switch (col) {
  case BT_COL_SERIES_KEY:
    sqlite3_result_text(ctx, r->series_key ? r->series_key : "", -1,
                        SQLITE_TRANSIENT);
    break;
  case BT_COL_FOLD:
    if (r->has_metrics)
      sqlite3_result_int(ctx, r->fold);
    break;
  case BT_COL_CUTOFF:
    if (r->has_metrics)
      sqlite3_result_text(ctx, r->cutoff, -1, SQLITE_TRANSIENT);
    break;
  case BT_COL_MODEL:
    if (r->has_metrics)
      sqlite3_result_text(ctx, r->model, -1, SQLITE_STATIC);
    break;
  case BT_COL_N:
    if (r->has_metrics)
      sqlite3_result_int(ctx, r->n);
    break;
  case BT_COL_MAE:
    if (r->has_metrics)
      sqlite3_result_double(ctx, r->mae);
    break;
  case BT_COL_RMSE:
    if (r->has_metrics)
      sqlite3_result_double(ctx, r->rmse);
    break;
  case BT_COL_MASE:
    if (r->has_metrics)
      sqlite3_result_double(ctx, r->mase);
    break;
  case BT_COL_SMAPE:
    if (r->has_metrics)
      sqlite3_result_double(ctx, r->smape);
    break;
  case BT_COL_COVERAGE:
    if (r->has_band)
      sqlite3_result_double(ctx, r->coverage);
    break;
  case BT_COL_WIDTH:
    if (r->has_band)
      sqlite3_result_double(ctx, r->width);
    break;
  case BT_COL_STATUS:
    sqlite3_result_text(ctx, r->status, -1, SQLITE_STATIC);
    break;
  case BT_COL_RECEIPT:
    if (r->receipt_id[0])
      sqlite3_result_text(ctx, r->receipt_id, -1, SQLITE_TRANSIENT);
    break;
  default:
    break;
  }
  return SQLITE_OK;
}

static int bt_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  *pRowid = ((bt_cursor *)pCur)->i;
  return SQLITE_OK;
}

static sqlite3_module backtestModule = {
    /* iVersion    */ 0,
    /* xCreate     */ NULL,
    /* xConnect    */ bt_connect,
    /* xBestIndex  */ bt_best_index,
    /* xDisconnect */ bt_disconnect,
    /* xDestroy    */ NULL,
    /* xOpen       */ bt_open,
    /* xClose      */ bt_close,
    /* xFilter     */ bt_filter,
    /* xNext       */ bt_next,
    /* xEof        */ bt_eof,
    /* xColumn     */ bt_column,
    /* xRowid      */ bt_rowid,
    /* xUpdate     */ NULL,
    /* xBegin      */ NULL,
    /* xSync       */ NULL,
    /* xCommit     */ NULL,
    /* xRollback   */ NULL,
    /* xFindMethod */ NULL,
    /* xRename     */ NULL,
    /* xSavepoint  */ NULL,
    /* xRelease    */ NULL,
    /* xRollbackTo */ NULL,
    /* xShadowName */ NULL,
    /* xIntegrity  */ NULL};

#pragma endregion

#pragma region aggregate

/* forecast(ts, value, horizon[, options]) and detect_anomalies(ts,
 * value[, options]) — the aggregate forms, RFC §4.2.8. The enclosing
 * statement pushes rows in (GROUP BY = series splitting), the model
 * runs through the same fc_/an_ pipeline as the query form, and the
 * result is one JSON document per group. Receipts embed the input
 * series ('inline-series', §4.1.4) and are written only for series the
 * model actually served; degraded series (insufficient_history,
 * non_numeric) return a status document with receipt_id null. */

/* aggregate option keys: the query form's set minus the query-shape
 * keys (time_col/value_col/group_cols), which GROUP BY and the argument
 * positions replace */
static const char *const AGG_FORECAST_OPTION_KEYS[] = {
    "confidence_level", "context_limit", "model",      "receipt",
    "interval_method",  "folds",         "gap",        "candidates", NULL};
static const char *const AGG_ANOMALY_OPTION_KEYS[] = {
    "anomaly_prob_threshold", "context_limit", "model", "receipt", NULL};

typedef struct {
  i64 *ts;
  f64 *val;
  int n, cap;
  int non_numeric; /* a row failed coercion; degrade like the query form */
  char *options_json; /* first row's options text; NULL = no options */
  int have_options;
  i64 horizon; /* forecast only */
  int have_horizon;
  int started;
} AggSeries;

static void agg_series_clear(AggSeries *a) {
  if (!a)
    return;
  sqlite3_free(a->ts);
  sqlite3_free(a->val);
  sqlite3_free(a->options_json);
  a->ts = NULL;
  a->val = NULL;
  a->options_json = NULL;
}

/* TEXT first argument that reads as SQL: the caller invoked the query
 * form in expression position; redirect them (RFC §4.2.8 Errors). */
static int agg_looks_like_query(sqlite3_value *v) {
  if (sqlite3_value_type(v) != SQLITE_TEXT)
    return 0;
  const char *t = (const char *)sqlite3_value_text(v);
  if (!t)
    return 0;
  while (*t == ' ' || *t == '\t' || *t == '\n' || *t == '\r')
    t++;
  return sqlite3_strnicmp(t, "select", 6) == 0 ||
         sqlite3_strnicmp(t, "with", 4) == 0;
}

/* Shared xStep: coerce (ts, value) with the query form's rules
 * (collect_series), enforce constant options (and horizon, when
 * horizon_arg >= 0), and append. Timestamps normalize to whole seconds
 * so the model reads exactly what the receipt records (§4.2.8). */
static void agg_step(sqlite3_context *ctx, int argc, sqlite3_value **argv,
                     const char *fname, int horizon_arg) {
  AggSeries *a = sqlite3_aggregate_context(ctx, sizeof(AggSeries));
  if (!a) {
    sqlite3_result_error_nomem(ctx);
    return;
  }

  if (!a->started) {
    a->started = 1;
    if (agg_looks_like_query(argv[0])) {
      char *msg = sqlite3_mprintf(
          "%s: %s(query, ...) is the table-valued form and belongs in the "
          "FROM clause; in expression position use the aggregate "
          "%s(ts, value%s[, options])",
          PREDICT_ERR_SCHEMA, fname, fname, horizon_arg >= 0 ? ", horizon" : "");
      sqlite3_result_error(ctx, msg ? msg : PREDICT_ERR_SCHEMA, -1);
      sqlite3_free(msg);
      return;
    }
    if (horizon_arg >= 0) {
      if (sqlite3_value_type(argv[horizon_arg]) != SQLITE_INTEGER) {
        sqlite3_result_error(
            ctx, PREDICT_ERR_HORIZON ": horizon must be an integer", -1);
        return;
      }
      a->horizon = sqlite3_value_int64(argv[horizon_arg]);
      a->have_horizon = 1;
      if (a->horizon < 1 || a->horizon > FORECAST_MAX_HORIZON) {
        sqlite3_result_error(
            ctx, PREDICT_ERR_HORIZON ": horizon out of range 1..1000", -1);
        return;
      }
    }
    int opt_i = horizon_arg >= 0 ? horizon_arg + 1 : 2;
    if (argc > opt_i && sqlite3_value_type(argv[opt_i]) != SQLITE_NULL) {
      const char *o = (const char *)sqlite3_value_text(argv[opt_i]);
      a->options_json = o ? sqlite3_mprintf("%s", o) : NULL;
      if (o && !a->options_json) {
        sqlite3_result_error_nomem(ctx);
        return;
      }
    }
    a->have_options = 1;
  } else {
    /* options (and horizon) MUST be constant across the group */
    if (horizon_arg >= 0 &&
        (sqlite3_value_type(argv[horizon_arg]) != SQLITE_INTEGER ||
         sqlite3_value_int64(argv[horizon_arg]) != a->horizon)) {
      sqlite3_result_error(
          ctx, PREDICT_ERR_HORIZON ": horizon must be constant within a group",
          -1);
      return;
    }
    int opt_i = horizon_arg >= 0 ? horizon_arg + 1 : 2;
    const char *o =
        argc > opt_i && sqlite3_value_type(argv[opt_i]) != SQLITE_NULL
            ? (const char *)sqlite3_value_text(argv[opt_i])
            : NULL;
    int differ = (o == NULL) != (a->options_json == NULL) ||
                 (o && strcmp(o, a->options_json) != 0);
    if (differ) {
      sqlite3_result_error(
          ctx, PREDICT_ERR_OPTIONS ": options must be constant within a group",
          -1);
      return;
    }
  }

  if (a->non_numeric)
    return; /* series poisoned; keep consuming rows, like collect_series */

  /* ts coercion, mirroring collect_series */
  i64 ms = 0;
  int tt = sqlite3_value_type(argv[0]);
  if (tt == SQLITE_INTEGER) {
    i64 raw = sqlite3_value_int64(argv[0]);
    if (raw < 0) {
      a->non_numeric = 1;
      return;
    }
    ms = raw >= PREDICT_EPOCH_MS_THRESHOLD ? raw : raw * 1000;
    if (ms > PREDICT_MS_MAX) {
      a->non_numeric = 1;
      return;
    }
  } else if (tt != SQLITE_TEXT ||
             predict0_parse_timestamp(
                 (const char *)sqlite3_value_text(argv[0]), &ms)) {
    a->non_numeric = 1;
    return;
  }
  ms -= ((ms % 1000) + 1000) % 1000; /* floor to whole seconds */

  int vt = sqlite3_value_type(argv[1]);
  if (vt != SQLITE_INTEGER && vt != SQLITE_FLOAT) {
    a->non_numeric = 1;
    return;
  }

  if (a->n >= FORECAST_MAX_TOTAL_ROWS) {
    sqlite3_result_error(ctx, PREDICT_ERR_RESOURCE ": too many input rows",
                         -1);
    return;
  }
  if (a->n == a->cap) {
    int nc = a->cap ? a->cap * 2 : 64;
    i64 *nts = sqlite3_realloc(a->ts, sizeof(i64) * nc);
    f64 *nval = sqlite3_realloc(a->val, sizeof(f64) * nc);
    if (!nts || !nval) {
      sqlite3_free(nts);
      sqlite3_free(nval);
      a->ts = NULL;
      a->val = NULL;
      a->cap = 0;
      a->n = 0;
      sqlite3_result_error_nomem(ctx);
      return;
    }
    a->ts = nts;
    a->val = nval;
    a->cap = nc;
  }
  a->ts[a->n] = ms;
  a->val[a->n] = sqlite3_value_double(argv[1]);
  a->n++;
}

static void agg_fc_step(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  agg_step(ctx, argc, argv, "forecast", 2);
}

static void agg_an_step(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  agg_step(ctx, argc, argv, "detect_anomalies", -1);
}

/* the query form called in expression position at its own arity:
 * turn the likeliest confusion into directions (RFC §4.2.8 Errors) */
static void agg_fc_misuse(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  sqlite3_result_error(
      ctx,
      PREDICT_ERR_SCHEMA
      ": forecast(query, horizon) is the table-valued form and belongs in "
      "the FROM clause; in expression position use the aggregate "
      "forecast(ts, value, horizon[, options])",
      -1);
}

/* JSON emission: %.17g round-trips doubles exactly (SQLite's own JSON
 * REAL formatting does not), which the inline-series replay hash
 * depends on. Non-finite values have no JSON form and emit null. */
static void agg_json_num(sqlite3_str *s, f64 v) {
  if (!isfinite(v)) {
    sqlite3_str_appendall(s, "null");
    return;
  }
  char buf[40];
  snprintf(buf, sizeof(buf), "%.17g", v);
  sqlite3_str_appendall(s, buf);
}

static void agg_json_str(sqlite3_str *s, const char *t) {
  sqlite3_str_appendchar(s, 1, '"');
  for (; *t; t++) {
    unsigned char c = (unsigned char)*t;
    if (c == '"' || c == '\\') {
      sqlite3_str_appendchar(s, 1, '\\');
      sqlite3_str_appendchar(s, 1, (char)c);
    } else if (c < 0x20) {
      sqlite3_str_appendf(s, "\\u%04x", c);
    } else {
      sqlite3_str_appendchar(s, 1, (char)c);
    }
  }
  sqlite3_str_appendchar(s, 1, '"');
}

/* canonical input-series JSON + its §4.1.4 digest over the rows the
 * model actually read (post-sort, post-truncation) */
static char *agg_input_data(const i64 *ts, const f64 *val, int start, int n,
                            char digest_out[PREDICT_HEX_BUFSIZE]) {
  sqlite3_str *s = sqlite3_str_new(NULL);
  predict0_hasher h;
  predict0_hash_init(&h);
  sqlite3_str_appendall(s, "{\"ts\":[");
  char iso[PREDICT_TS_BUFSIZE];
  for (int i = start; i < n; i++) {
    predict0_format_timestamp(ts[i], iso, sizeof(iso));
    sqlite3_str_appendf(s, "%s\"%s\"", i > start ? "," : "", iso);
    predict0_hash_text(&h, iso);
    predict0_hash_real(&h, val[i]);
    predict0_hash_row_end(&h);
  }
  sqlite3_str_appendall(s, "],\"value\":[");
  for (int i = start; i < n; i++) {
    if (i > start)
      sqlite3_str_appendchar(s, 1, ',');
    agg_json_num(s, val[i]);
  }
  sqlite3_str_appendall(s, "]}");
  predict0_hash_hex(&h, digest_out);
  return sqlite3_str_finish(s); /* NULL on OOM */
}

/* result_error from a "CODE: detail" sqlite3_malloc'd message */
static void agg_error(sqlite3_context *ctx, char *msg) {
  sqlite3_result_error(ctx, msg ? msg : "PREDICT_ERR_RESOURCE: out of memory",
                       -1);
  sqlite3_free(msg);
}

static void agg_fc_final(sqlite3_context *ctx) {
  AggSeries *a = sqlite3_aggregate_context(ctx, 0);
  if (!a || !a->started) {
    /* zero rows: NULL, the SQL aggregate convention (§4.2.8) */
    agg_series_clear(a);
    sqlite3_result_null(ctx);
    return;
  }
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  int horizon = (int)a->horizon;

  ForecastOpts opts;
  memset(&opts, 0, sizeof(opts));
  opts.confidence = 0.95;
  opts.context_limit = FORECAST_DEFAULT_CONTEXT_LIMIT;
  opts.receipt = 1;
  char *perr = NULL;
  if (predict0_options_parse(db, a->options_json, AGG_FORECAST_OPTION_KEYS,
                             forecast_opt_cb, &opts, &perr)) {
    agg_error(ctx, perr);
    forecast_opts_free(&opts);
    agg_series_clear(a);
    return;
  }

  FcModels mc;
  char *mrerr = NULL;
  if (fc_resolve_models(db, &opts, horizon, &mc, &mrerr) != SQLITE_OK) {
    agg_error(ctx, mrerr);
    forecast_opts_free(&opts);
    agg_series_clear(a);
    return;
  }
  const char *model_id = fc_model_id(&opts, &mc);

  SeriesBuf s;
  memset(&s, 0, sizeof(s));
  s.key = "";
  s.ts = a->ts;
  s.val = a->val;
  s.n = a->n;
  s.cap = a->cap;
  s.non_numeric = a->non_numeric;

  ForecastRow *rows = NULL;
  int n_rows = 0;
  char *serve_err = NULL;
  int rc = fc_run_series(db, &s, 1, horizon, &opts, &mc, &rows, &n_rows,
                         &serve_err);
  fc_models_free(&mc);
  if (rc != SQLITE_OK) {
    if (serve_err)
      agg_error(ctx, serve_err);
    else
      sqlite3_result_error_nomem(ctx);
    forecast_opts_free(&opts);
    agg_series_clear(a);
    return;
  }

  /* one series: status is uniform across rows */
  const char *status = n_rows ? rows[0].status : "ok";
  int served = n_rows > 0 && rows[0].has_values;

  /* receipt: only for series the model actually served (§4.2.8) */
  char receipt_id[PREDICT_ULID_BUFSIZE] = "";
  if (opts.receipt && served) {
    char result_hash[PREDICT_HEX_BUFSIZE];
    char sdigest[PREDICT_HEX_BUFSIZE];
    char *input_data = NULL;
    char *params = NULL;
    char *rerr = NULL;
    int start = (opts.context_limit > 0 && a->n > opts.context_limit)
                    ? a->n - opts.context_limit
                    : 0;
    int irc = fc_result_hash(rows, n_rows, result_hash);
    if (irc == SQLITE_OK) {
      input_data = agg_input_data(a->ts, a->val, start, a->n, sdigest);
      params = fc_build_params(db, &opts, horizon, model_id, NULL, NULL, 1);
      if (!input_data || !params)
        irc = SQLITE_NOMEM;
    }
    if (irc == SQLITE_OK)
      irc = predict0_emit_receipt_inline(db, "forecast", model_id, params,
                                         input_data, sdigest, result_hash,
                                         receipt_id, &rerr);
    sqlite3_free(input_data);
    sqlite3_free(params);
    if (irc != SQLITE_OK) {
      agg_error(ctx, rerr);
      for (int i = 0; i < n_rows; i++)
        sqlite3_free(rows[i].series_key);
      sqlite3_free(rows);
      forecast_opts_free(&opts);
      agg_series_clear(a);
      return;
    }
  }

  /* the result document (§4.2.8): row fields named as the query form's
   * output columns */
  sqlite3_str *doc = sqlite3_str_new(db);
  sqlite3_str_appendall(doc, "{\"model\":");
  agg_json_str(doc, model_id);
  sqlite3_str_appendall(doc, ",\"receipt_id\":");
  if (receipt_id[0])
    sqlite3_str_appendf(doc, "\"%s\"", receipt_id);
  else
    sqlite3_str_appendall(doc, "null");
  sqlite3_str_appendf(doc, ",\"status\":\"%s\",\"rows\":[", status);
  int first = 1;
  for (int i = 0; i < n_rows; i++) {
    if (!rows[i].has_values)
      continue;
    if (!first)
      sqlite3_str_appendchar(doc, 1, ',');
    first = 0;
    sqlite3_str_appendf(doc, "{\"step\":%d,\"forecast_timestamp\":\"%s\","
                             "\"forecast\":",
                        rows[i].step, rows[i].ts);
    agg_json_num(doc, rows[i].forecast);
    sqlite3_str_appendall(doc, ",\"lower_bound\":");
    agg_json_num(doc, rows[i].lower);
    sqlite3_str_appendall(doc, ",\"upper_bound\":");
    agg_json_num(doc, rows[i].upper);
    sqlite3_str_appendchar(doc, 1, '}');
  }
  sqlite3_str_appendall(doc, "]}");

  for (int i = 0; i < n_rows; i++)
    sqlite3_free(rows[i].series_key);
  sqlite3_free(rows);
  forecast_opts_free(&opts);
  agg_series_clear(a);

  char *out = sqlite3_str_finish(doc);
  if (!out) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  sqlite3_result_text(ctx, out, -1, sqlite3_free);
}

static void agg_an_final(sqlite3_context *ctx) {
  AggSeries *a = sqlite3_aggregate_context(ctx, 0);
  if (!a || !a->started) {
    agg_series_clear(a);
    sqlite3_result_null(ctx);
    return;
  }
  sqlite3 *db = sqlite3_context_db_handle(ctx);

  ForecastOpts opts;
  memset(&opts, 0, sizeof(opts));
  opts.confidence = 0.99; /* anomaly_prob_threshold default */
  opts.context_limit = 0; /* score every point, the query form's default */
  opts.receipt = 1;
  char *perr = NULL;
  if (predict0_options_parse(db, a->options_json, AGG_ANOMALY_OPTION_KEYS,
                             forecast_opt_cb, &opts, &perr)) {
    agg_error(ctx, perr);
    forecast_opts_free(&opts);
    agg_series_clear(a);
    return;
  }

  int use_subpca = 0, theta_mode = 1;
  const char *model_id = NULL;
  char *mrerr = NULL;
  if (an_resolve_model(&opts, &use_subpca, &theta_mode, &model_id, &mrerr) !=
      SQLITE_OK) {
    agg_error(ctx, mrerr);
    forecast_opts_free(&opts);
    agg_series_clear(a);
    return;
  }

  SeriesBuf s;
  memset(&s, 0, sizeof(s));
  s.key = "";
  s.ts = a->ts;
  s.val = a->val;
  s.n = a->n;
  s.cap = a->cap;
  s.non_numeric = a->non_numeric;

  AnomRow *rows = NULL;
  int n_rows = 0;
  int rc = an_run_series(&s, 1, &opts, use_subpca, theta_mode, &rows, &n_rows);
  if (rc != SQLITE_OK) {
    sqlite3_result_error_nomem(ctx);
    forecast_opts_free(&opts);
    agg_series_clear(a);
    return;
  }

  const char *status = n_rows ? rows[0].status : "ok";
  int served = n_rows > 0 && rows[0].has_values;

  char receipt_id[PREDICT_ULID_BUFSIZE] = "";
  if (opts.receipt && served) {
    char result_hash[PREDICT_HEX_BUFSIZE];
    char sdigest[PREDICT_HEX_BUFSIZE];
    char *input_data = NULL;
    char *params = NULL;
    char *rerr = NULL;
    int start = (opts.context_limit > 0 && a->n > opts.context_limit)
                    ? a->n - opts.context_limit
                    : 0;
    int irc = an_result_hash(rows, n_rows, result_hash);
    if (irc == SQLITE_OK) {
      input_data = agg_input_data(a->ts, a->val, start, a->n, sdigest);
      params = an_build_params(db, &opts, model_id, NULL, NULL, 1);
      if (!input_data || !params)
        irc = SQLITE_NOMEM;
    }
    if (irc == SQLITE_OK)
      irc = predict0_emit_receipt_inline(db, "detect_anomalies", model_id,
                                         params, input_data, sdigest,
                                         result_hash, receipt_id, &rerr);
    sqlite3_free(input_data);
    sqlite3_free(params);
    if (irc != SQLITE_OK) {
      agg_error(ctx, rerr);
      for (int i = 0; i < n_rows; i++)
        sqlite3_free(rows[i].series_key);
      sqlite3_free(rows);
      forecast_opts_free(&opts);
      agg_series_clear(a);
      return;
    }
  }

  sqlite3_str *doc = sqlite3_str_new(db);
  sqlite3_str_appendall(doc, "{\"model\":");
  agg_json_str(doc, model_id);
  sqlite3_str_appendall(doc, ",\"receipt_id\":");
  if (receipt_id[0])
    sqlite3_str_appendf(doc, "\"%s\"", receipt_id);
  else
    sqlite3_str_appendall(doc, "null");
  sqlite3_str_appendf(doc, ",\"status\":\"%s\",\"rows\":[", status);
  int first = 1;
  for (int i = 0; i < n_rows; i++) {
    if (!rows[i].has_values)
      continue;
    if (!first)
      sqlite3_str_appendchar(doc, 1, ',');
    first = 0;
    sqlite3_str_appendf(doc, "{\"ts\":\"%s\",\"value\":", rows[i].ts);
    agg_json_num(doc, rows[i].value);
    sqlite3_str_appendall(doc, ",\"forecast\":");
    if (rows[i].has_pred)
      agg_json_num(doc, rows[i].fc);
    else
      sqlite3_str_appendall(doc, "null");
    sqlite3_str_appendall(doc, ",\"lower_bound\":");
    if (rows[i].has_pred)
      agg_json_num(doc, rows[i].lo);
    else
      sqlite3_str_appendall(doc, "null");
    sqlite3_str_appendall(doc, ",\"upper_bound\":");
    if (rows[i].has_pred)
      agg_json_num(doc, rows[i].hi);
    else
      sqlite3_str_appendall(doc, "null");
    sqlite3_str_appendf(doc, ",\"is_anomaly\":%d,\"anomaly_probability\":",
                        rows[i].is_anom);
    agg_json_num(doc, rows[i].prob);
    sqlite3_str_appendchar(doc, 1, '}');
  }
  sqlite3_str_appendall(doc, "]}");

  for (int i = 0; i < n_rows; i++)
    sqlite3_free(rows[i].series_key);
  sqlite3_free(rows);
  forecast_opts_free(&opts);
  agg_series_clear(a);

  char *out = sqlite3_str_finish(doc);
  if (!out) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  sqlite3_result_text(ctx, out, -1, sqlite3_free);
}

#pragma endregion

#pragma region expansion

/* forecast_rows(doc) / anomaly_rows(doc) — expand an aggregate-form
 * JSON document back into typed rows (RFC §4.2.9). Column names and
 * types match the query form's output columns exactly; the numeric
 * columns are forced REAL so the inline-series replay hash sees the
 * same encoding the write path produced. */

typedef struct {
  i64 ints[2];   /* step | is_anomaly slots */
  f64 reals[5];  /* forecast/lower/upper (+ value/prob for anomaly) */
  char *ts;      /* forecast_timestamp | ts */
  u8 has_int[2], has_real[5], has_ts;
} XRow;

typedef struct {
  sqlite3_vtab base;
  sqlite3 *db;
  int is_anom; /* 0 = forecast_rows, 1 = anomaly_rows */
} xrows_vtab;

typedef struct {
  sqlite3_vtab_cursor base;
  XRow *rows;
  int n_rows;
  int i;
  char *status;
  char *receipt_id;
} xrows_cursor;

static int xr_connect(sqlite3 *db, void *pAux, int argc,
                      const char *const *argv, sqlite3_vtab **ppVtab,
                      char **pzErr) {
  UNUSED_PARAMETER(argc);
  UNUSED_PARAMETER(argv);
  UNUSED_PARAMETER(pzErr);
  int is_anom = pAux != NULL;
  xrows_vtab *v = sqlite3_malloc(sizeof(*v));
  if (!v)
    return SQLITE_NOMEM;
  memset(v, 0, sizeof(*v));
  v->db = db;
  v->is_anom = is_anom;
  int rc = sqlite3_declare_vtab(
      db, is_anom
              ? "CREATE TABLE x(ts TEXT, value REAL, forecast REAL,"
                " lower_bound REAL, upper_bound REAL, is_anomaly INTEGER,"
                " anomaly_probability REAL, status TEXT, receipt_id TEXT,"
                " doc HIDDEN)"
              : "CREATE TABLE x(step INTEGER, forecast_timestamp TEXT,"
                " forecast REAL, lower_bound REAL, upper_bound REAL,"
                " status TEXT, receipt_id TEXT, doc HIDDEN)");
  if (rc != SQLITE_OK) {
    sqlite3_free(v);
    return rc;
  }
  *ppVtab = &v->base;
  return SQLITE_OK;
}

static int xr_disconnect(sqlite3_vtab *pVtab) {
  sqlite3_free(pVtab);
  return SQLITE_OK;
}

static int xr_best_index(sqlite3_vtab *pVtab, sqlite3_index_info *pIdx) {
  xrows_vtab *v = (xrows_vtab *)pVtab;
  int doc_col = v->is_anom ? 9 : 7;
  int seen = 0;
  for (int i = 0; i < pIdx->nConstraint; i++) {
    const struct sqlite3_index_constraint *c = &pIdx->aConstraint[i];
    if (c->iColumn == doc_col && c->op == SQLITE_INDEX_CONSTRAINT_EQ) {
      if (!c->usable)
        return SQLITE_CONSTRAINT;
      pIdx->aConstraintUsage[i].argvIndex = 1;
      pIdx->aConstraintUsage[i].omit = 1;
      seen = 1;
    }
  }
  if (!seen) {
    pVtab->zErrMsg = sqlite3_mprintf(
        "%s: %s(doc) requires a document argument", PREDICT_ERR_SCHEMA,
        v->is_anom ? "anomaly_rows" : "forecast_rows");
    return SQLITE_ERROR;
  }
  pIdx->estimatedCost = 1000;
  return SQLITE_OK;
}

static int xr_open(sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCur) {
  UNUSED_PARAMETER(pVtab);
  xrows_cursor *c = sqlite3_malloc(sizeof(*c));
  if (!c)
    return SQLITE_NOMEM;
  memset(c, 0, sizeof(*c));
  *ppCur = &c->base;
  return SQLITE_OK;
}

static void xr_rows_free(xrows_cursor *c) {
  for (int i = 0; i < c->n_rows; i++)
    sqlite3_free(c->rows[i].ts);
  sqlite3_free(c->rows);
  sqlite3_free(c->status);
  sqlite3_free(c->receipt_id);
  c->rows = NULL;
  c->n_rows = 0;
  c->status = NULL;
  c->receipt_id = NULL;
  c->i = 0;
}

static int xr_close(sqlite3_vtab_cursor *pCur) {
  xrows_cursor *c = (xrows_cursor *)pCur;
  xr_rows_free(c);
  sqlite3_free(c);
  return SQLITE_OK;
}

static int xr_error(xrows_cursor *cur, const char *detail) {
  sqlite3_free(cur->base.pVtab->zErrMsg);
  cur->base.pVtab->zErrMsg =
      sqlite3_mprintf("%s: %s", PREDICT_ERR_SCHEMA, detail);
  return SQLITE_ERROR;
}

static int xr_filter(sqlite3_vtab_cursor *pCur, int idxNum, const char *idxStr,
                     int argc, sqlite3_value **argv) {
  UNUSED_PARAMETER(idxNum);
  UNUSED_PARAMETER(idxStr);
  xrows_cursor *cur = (xrows_cursor *)pCur;
  xrows_vtab *vtab = (xrows_vtab *)cur->base.pVtab;
  sqlite3 *db = vtab->db;
  xr_rows_free(cur);

  if (argc < 1 || sqlite3_value_type(argv[0]) == SQLITE_NULL)
    return SQLITE_OK; /* NULL in, empty out (the json_each convention) */
  const char *doc = (const char *)sqlite3_value_text(argv[0]);
  if (!doc)
    return xr_error(cur, "document must be text");

  /* validate the document shape and lift the top-level fields */
  sqlite3_stmt *top = NULL;
  if (sqlite3_prepare_v2(db,
                         "SELECT json_type(?1), json_type(?1,'$.rows'),"
                         " json_extract(?1,'$.status'),"
                         " json_extract(?1,'$.receipt_id')",
                         -1, &top, NULL) != SQLITE_OK)
    return xr_error(cur, "could not parse document");
  sqlite3_bind_text(top, 1, doc, -1, SQLITE_STATIC);
  int rc = sqlite3_step(top);
  const char *jt = rc == SQLITE_ROW ? (const char *)sqlite3_column_text(top, 0)
                                    : NULL;
  const char *rt = rc == SQLITE_ROW ? (const char *)sqlite3_column_text(top, 1)
                                    : NULL;
  if (!jt || strcmp(jt, "object") != 0 || !rt || strcmp(rt, "array") != 0 ||
      sqlite3_column_type(top, 2) != SQLITE_TEXT) {
    sqlite3_finalize(top);
    return xr_error(cur, "not an aggregate-form result document");
  }
  cur->status =
      sqlite3_mprintf("%s", (const char *)sqlite3_column_text(top, 2));
  if (sqlite3_column_type(top, 3) == SQLITE_TEXT)
    cur->receipt_id =
        sqlite3_mprintf("%s", (const char *)sqlite3_column_text(top, 3));
  sqlite3_finalize(top);

  const char *rows_sql =
      vtab->is_anom
          ? "SELECT json_extract(j.value,'$.ts'),"
            " json_extract(j.value,'$.value'),"
            " json_extract(j.value,'$.forecast'),"
            " json_extract(j.value,'$.lower_bound'),"
            " json_extract(j.value,'$.upper_bound'),"
            " json_extract(j.value,'$.is_anomaly'),"
            " json_extract(j.value,'$.anomaly_probability')"
            " FROM json_each(?1,'$.rows') j ORDER BY j.key"
          : "SELECT json_extract(j.value,'$.step'),"
            " json_extract(j.value,'$.forecast_timestamp'),"
            " json_extract(j.value,'$.forecast'),"
            " json_extract(j.value,'$.lower_bound'),"
            " json_extract(j.value,'$.upper_bound')"
            " FROM json_each(?1,'$.rows') j ORDER BY j.key";
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db, rows_sql, -1, &stmt, NULL) != SQLITE_OK)
    return xr_error(cur, "could not parse document rows");
  sqlite3_bind_text(stmt, 1, doc, -1, SQLITE_STATIC);

  int cap = 0;
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    if (cur->n_rows == cap) {
      int nc = cap ? cap * 2 : 32;
      XRow *grown = sqlite3_realloc(cur->rows, sizeof(XRow) * nc);
      if (!grown) {
        sqlite3_finalize(stmt);
        return SQLITE_NOMEM;
      }
      cur->rows = grown;
      cap = nc;
    }
    XRow *r = &cur->rows[cur->n_rows++];
    memset(r, 0, sizeof(*r));
    if (vtab->is_anom) {
      /* ts, value, forecast, lower, upper, is_anomaly, probability */
      if (sqlite3_column_type(stmt, 0) == SQLITE_TEXT) {
        r->ts = sqlite3_mprintf("%s", sqlite3_column_text(stmt, 0));
        r->has_ts = 1;
      }
      static const int real_col[5] = {1, 2, 3, 4, 6};
      for (int k = 0; k < 5; k++)
        if (sqlite3_column_type(stmt, real_col[k]) != SQLITE_NULL) {
          r->reals[k] = sqlite3_column_double(stmt, real_col[k]);
          r->has_real[k] = 1;
        }
      if (sqlite3_column_type(stmt, 5) != SQLITE_NULL) {
        r->ints[1] = sqlite3_column_int64(stmt, 5);
        r->has_int[1] = 1;
      }
    } else {
      /* step, forecast_timestamp, forecast, lower, upper */
      if (sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
        r->ints[0] = sqlite3_column_int64(stmt, 0);
        r->has_int[0] = 1;
      }
      if (sqlite3_column_type(stmt, 1) == SQLITE_TEXT) {
        r->ts = sqlite3_mprintf("%s", sqlite3_column_text(stmt, 1));
        r->has_ts = 1;
      }
      for (int k = 0; k < 3; k++)
        if (sqlite3_column_type(stmt, 2 + k) != SQLITE_NULL) {
          r->reals[k] = sqlite3_column_double(stmt, 2 + k);
          r->has_real[k] = 1;
        }
    }
  }
  int done = rc == SQLITE_DONE;
  sqlite3_finalize(stmt);
  if (!done)
    return xr_error(cur, "document rows are not valid JSON");

  /* empty rows: one status row, the query form's convention (§4.2.9) */
  if (cur->n_rows == 0) {
    cur->rows = sqlite3_malloc(sizeof(XRow));
    if (!cur->rows)
      return SQLITE_NOMEM;
    memset(cur->rows, 0, sizeof(XRow));
    cur->n_rows = 1;
  }
  cur->i = 0;
  return SQLITE_OK;
}

static int xr_next(sqlite3_vtab_cursor *pCur) {
  ((xrows_cursor *)pCur)->i++;
  return SQLITE_OK;
}

static int xr_eof(sqlite3_vtab_cursor *pCur) {
  xrows_cursor *c = (xrows_cursor *)pCur;
  return c->i >= c->n_rows;
}

static int xr_column(sqlite3_vtab_cursor *pCur, sqlite3_context *ctx,
                     int col) {
  xrows_cursor *c = (xrows_cursor *)pCur;
  xrows_vtab *v = (xrows_vtab *)c->base.pVtab;
  XRow *r = &c->rows[c->i];
  int status_col = v->is_anom ? 7 : 5;
  if (col == status_col) {
    if (c->status)
      sqlite3_result_text(ctx, c->status, -1, SQLITE_TRANSIENT);
    return SQLITE_OK;
  }
  if (col == status_col + 1) {
    if (c->receipt_id)
      sqlite3_result_text(ctx, c->receipt_id, -1, SQLITE_TRANSIENT);
    return SQLITE_OK;
  }
  if (v->is_anom) {
    switch (col) {
    case 0: /* ts */
      if (r->has_ts)
        sqlite3_result_text(ctx, r->ts, -1, SQLITE_TRANSIENT);
      break;
    case 1: case 2: case 3: case 4: /* value..upper_bound */
      if (r->has_real[col - 1])
        sqlite3_result_double(ctx, r->reals[col - 1]);
      break;
    case 5: /* is_anomaly */
      if (r->has_int[1])
        sqlite3_result_int64(ctx, r->ints[1]);
      break;
    case 6: /* anomaly_probability */
      if (r->has_real[4])
        sqlite3_result_double(ctx, r->reals[4]);
      break;
    default:
      break;
    }
  } else {
    switch (col) {
    case 0: /* step */
      if (r->has_int[0])
        sqlite3_result_int64(ctx, r->ints[0]);
      break;
    case 1: /* forecast_timestamp */
      if (r->has_ts)
        sqlite3_result_text(ctx, r->ts, -1, SQLITE_TRANSIENT);
      break;
    case 2: case 3: case 4: /* forecast..upper_bound */
      if (r->has_real[col - 2])
        sqlite3_result_double(ctx, r->reals[col - 2]);
      break;
    default:
      break;
    }
  }
  return SQLITE_OK;
}

static int xr_rowid(sqlite3_vtab_cursor *pCur, sqlite3_int64 *pRowid) {
  *pRowid = ((xrows_cursor *)pCur)->i;
  return SQLITE_OK;
}

static sqlite3_module xrowsModule = {
    /* iVersion    */ 0,
    /* xCreate     */ NULL,
    /* xConnect    */ xr_connect,
    /* xBestIndex  */ xr_best_index,
    /* xDisconnect */ xr_disconnect,
    /* xDestroy    */ NULL,
    /* xOpen       */ xr_open,
    /* xClose      */ xr_close,
    /* xFilter     */ xr_filter,
    /* xNext       */ xr_next,
    /* xEof        */ xr_eof,
    /* xColumn     */ xr_column,
    /* xRowid      */ xr_rowid,
    /* xUpdate     */ NULL,
    /* xBegin      */ NULL,
    /* xSync       */ NULL,
    /* xCommit     */ NULL,
    /* xRollback   */ NULL,
    /* xFindMethod */ NULL,
    /* xRename     */ NULL,
    /* xSavepoint  */ NULL,
    /* xRelease    */ NULL,
    /* xRollbackTo */ NULL,
    /* xShadowName */ NULL,
    /* xIntegrity  */ NULL};

#pragma endregion

int predict0_forecast_init(sqlite3 *db) {
  int rc = sqlite3_create_module(db, "forecast", &forecastModule, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_module(db, "detect_anomalies", &anomModule, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_module(db, "backtest", &backtestModule, NULL);
  if (rc != SQLITE_OK)
    return rc;

  /* expansion functions (RFC §4.2.9); pAux discriminates the two shapes */
  rc = sqlite3_create_module(db, "forecast_rows", &xrowsModule, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_module(db, "anomaly_rows", &xrowsModule, (void *)1);
  if (rc != SQLITE_OK)
    return rc;

  /* Aggregate forms (RFC §4.2.8), same names as the table-valued forms:
   * SQLite keeps function and vtab-module namespaces separate, so
   * expression position resolves here and FROM position resolves above.
   * SQLITE_DIRECTONLY per §6.7 (receipt writes are a side effect), and
   * never SQLITE_DETERMINISTIC. */
  static const int AGG_FLAGS = SQLITE_UTF8 | SQLITE_DIRECTONLY;
  rc = sqlite3_create_function_v2(db, "forecast", 3, AGG_FLAGS, NULL, NULL,
                                  agg_fc_step, agg_fc_final, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_function_v2(db, "forecast", 4, AGG_FLAGS, NULL, NULL,
                                  agg_fc_step, agg_fc_final, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_function_v2(db, "detect_anomalies", 2, AGG_FLAGS, NULL,
                                  NULL, agg_an_step, agg_an_final, NULL);
  if (rc != SQLITE_OK)
    return rc;
  rc = sqlite3_create_function_v2(db, "detect_anomalies", 3, AGG_FLAGS, NULL,
                                  NULL, agg_an_step, agg_an_final, NULL);
  if (rc != SQLITE_OK)
    return rc;
  /* guidance stub: the query form's arity in expression position gets a
   * self-correcting error instead of "wrong number of arguments" */
  return sqlite3_create_function_v2(db, "forecast", 2, AGG_FLAGS, NULL,
                                    agg_fc_misuse, NULL, NULL, NULL);
}
