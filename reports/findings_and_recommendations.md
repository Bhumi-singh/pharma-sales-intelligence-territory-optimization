# Pharma Sales Analytics: Findings & Recommendations

## Executive Summary

Analysis of 254,082 transactions (2017-2020) across Germany and Poland reveals that
performance gaps between countries are driven primarily by **market-level dynamics**,
not sales execution. Germany generates ~16x more revenue than Poland, and country is
the single strongest predictor of transaction value in our model - more influential
than channel, product mix, or sales team. Both markets' core retail businesses are
forecast to decline over the next 12 months, while institutional/bulk revenue (38%
of total) is event-driven and should be managed via deal pipeline tracking rather
than statistical forecasting.

---

## Key Findings

### 1. Revenue Concentration
- 1% of transactions (bulk/institutional orders, quantity >= 1,600 units) generate
  **37.6% of total revenue** (~4.4B), concentrated among a handful of distributors
  (Gerlach LLC, Koss, Kozey-Emmerich, Bashirian-Kassulke).
- The remaining 99% of transactions (retail) generate 62.4% (~7.4B).

### 2. Country-Level Disparity
- Germany: ~11.26B total sales, 751 customers, 13 reps, 4 teams.
- Poland: ~0.68B total sales, 200 customers, same 13 reps, same 4 teams.
- Sales-per-rep: Germany ~866M vs Poland ~52M (16.5x gap).
- Transactions-per-rep: Germany ~16,229 vs Poland ~3,112 (5.2x gap) - Poland
  receives disproportionately less sales activity even relative to its smaller size.

### 3. Customer Segmentation (K-Means, k=4)
| Segment | Customers | % of Revenue | Growth (2017-2020) | Notes |
|---|---|---|---|---|
| Germany - Stable Key Accounts | 228 | ~47% | 3% | Highest avg transaction value; flat growth |
| Germany - Core Growth Retail | 309 | 43% | 11% | Largest segment; healthy growth |
| Germany - Rising Star Accounts | 14 | 4.25% | 515% | Tiny group, explosive growth - fast-track to key accounts |
| Poland - Underserved Market | 200 | 5.7% | 0% | Stagnant; lowest per-customer value |

### 4. Team Performance
- Team Delta leads overall sales (~3.6B) but has the **lowest** exposure to
  high-growth Rising Star accounts (2.82% vs 3.96-5.97% for other teams) -
  Delta outperforms via execution in core/stable segments, not favorable
  account assignment.
- Team-segment exposure is nearly uniform across Alfa/Bravo/Charlie/Delta
  (~42-47% Core/Stable, ~3-6% Rising Star, ~5.6-6% Poland) - performance gaps
  are execution-based, not portfolio-based.
- Team Charlie has the highest Rising Star exposure (5.97%) but middling
  overall sales - potential missed opportunity to convert high-growth accounts.

### 5. Forecasting (12-month outlook)
| Segment | Last 12mo Actual | Next 12mo Forecast | Change | Accuracy (MAPE) |
|---|---|---|---|---|
| Germany - Retail | 1.75B | 1.43B | -18.3% | 27.9% (reasonable) |
| Germany - Bulk | 0.97B | 1.40B | +44.4% | 65.2% (low confidence) |
| Poland - Retail | 0.60B | 0.46B | -23.9% | 18.5% (reasonable) |
| Poland - Bulk | 0.077B | 0.27B | +256.7% | 147.9% (not reliable) |

- Both core retail markets (62% of revenue) trend downward.
- Bulk forecasts have high error due to lumpy, deal-driven nature - should be
  tracked via CRM/pipeline, not time-series models.

### 6. Predictive Model & SHAP
- XGBoost model (excluding quantity) achieves R2=0.25 - low overall predictive
  power is expected without transaction size, but SHAP reveals **systematic
  drivers**:
  1. **country** (SHAP impact ~9,377) - by far the dominant factor
  2. month_num (~6,356) - seasonality
  3. product_class (~5,492)
  4. year (~4,846) - time trend
  5. sub_channel (~3,203)
  6. manager (~2,743)
  7. sales_team (~2,264)
  8. channel (~1,822)
- Poland transactions are systematically lower-value than Germany,
  **independent of channel, team, or product mix** - confirming the
  performance gap is market-level, not execution-level.

---

## Recommendations

### Priority 1: Address Poland's Under-Resourcing
- Poland receives the same rep/team coverage as Germany despite being 16x
  smaller, yet gets 5.2x fewer transactions per rep - increase visit frequency
  / activity targets in Poland to match Germany's intensity, then measure
  whether revenue-per-rep improves.
- If activity increases don't move the needle, treat Poland's low transaction
  value as a market-development issue (pricing, product availability,
  market access) rather than a coverage issue - the SHAP analysis suggests
  this is the more likely root cause.

### Priority 2: Reverse Germany-Retail Decline
- Germany-Retail (largest, most reliable segment, ~144M/month average) is
  forecast to decline ~18% - investigate causes (competitive pressure,
  pricing, product mix shifts by product_class and month, since both rank
  highly in SHAP).
- Cross-reference the declining months/product classes with the SHAP
  month_num and product_class impacts to identify where the decline is
  concentrated.

### Priority 3: Fast-Track Rising Star Accounts
- The 14 "Rising Star" customers (515% growth, ~94K avg transaction value)
  should be moved into dedicated key-account management before the
  relationship matures elsewhere. Team Charlie, with the highest exposure to
  this segment, should be evaluated for best practices or additional support
  in converting this opportunity.

### Priority 4: Separate Bulk and Retail Management
- Bulk/institutional revenue (38% of total, MAPE 65-148%) should be tracked
  via a deal pipeline (CRM-based, deal-by-deal) rather than statistical
  forecasting. Retail revenue (62% of total, MAPE 18-28%) can support more
  traditional forecasting and target-setting.

### Priority 5: Share Best Practices from Team Delta
- Team Delta's lead comes from execution in core/stable segments rather than
  favorable account mix (team-segment exposure is nearly identical across
  teams). Document and share Delta's approach to core-segment account
  management with Alfa, Bravo, and Charlie.

---

## Suggested Next Steps for Stakeholders
1. Pilot increased rep activity in Poland for one quarter; measure
   sales-per-rep and transactions-per-rep before/after.
2. Deep-dive into Germany-Retail decline by product_class and month to
   identify specific products/seasons driving the forecasted drop.
3. Establish a formal key-account onboarding process for Rising Star accounts.
4. Set up separate reporting cadences: retail (monthly forecast-based targets)
   vs bulk (pipeline/deal-based tracking).
