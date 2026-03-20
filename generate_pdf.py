"""
generate_pdf.py
AlgoGators rough draft - matches the official template exactly.

Template specs (from Rough Draft [TEMPLATE - DO NOT MODIFY].docx):
  - Heading color:  #2F5496
  - Heading 1:      16 pt
  - Heading 2:      13 pt
  - Body:           12 pt (Calibri - approximated with Helvetica)
  - Margins:        1 inch (25.4 mm) all sides
  - Header:         AlgoGators logo every page (except title)
  - Footer:         Centered page number, starts at 2
  - Sections:       1 Abstract  2 Introduction  3 Methodology
                    4 Results   5 Discussion    6 Conclusion
                    7 References  8 Appendices  + Disclaimer
"""

import os, sys
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image as PILImage

# ── constants ────────────────────────────────────────────────────────────────
TITLE      = "Regime-Conditional Tail Risk Modeling for Portfolio Risk Management"
AUTHOR     = "Nathan Hoang"
ORG        = "AlgoGators Capstone Project"
DATE       = "March 2026"
GITHUB_URL = "https://github.com/minh-stakc/algogator-tail-risk"
OUT        = "rough_draft.pdf"

# Template palette
HEAD_COL = (47, 84, 150)    # #2F5496 - all section headings
SUB_COL  = (47, 84, 150)    # same for sub-headings
BLACK    = (30, 30, 30)
GRAY     = (110, 110, 110)
LGRAY    = (210, 210, 210)
WHITE    = (255, 255, 255)
ROW_A    = (235, 241, 251)   # alternating table row tint
ROW_B    = (255, 255, 255)
TBL_HDR  = (47, 84, 150)     # table header = heading colour

# Page geometry
MARGIN   = 25.4              # 1 inch
PAGE_W   = 215.9             # Letter width mm
PAGE_H   = 279.4             # Letter height mm
USABLE_W = PAGE_W - 2*MARGIN # 165.1 mm
LOGO_W   = 48.0              # mm - logo rendered width in header

LOGO_FILE = "algogators_logo.png"


# ── content ──────────────────────────────────────────────────────────────────

ABSTRACT_TEXT = (
    "This paper investigates whether regime-aware risk models can produce "
    "better Conditional Value-at-Risk (CVaR) forecasts than the standard "
    "approaches portfolio managers typically use. Using daily returns from "
    "the S&P 500 and nine sector ETFs over 2005-2024, I detect market regimes "
    "with a Hidden Markov Model (HMM) and estimate tail risk within each "
    "regime using Peaks-over-Threshold extreme value theory. The results are "
    "mixed. Regime conditioning reveals something economically real - the "
    "high-volatility regime has a CVaR nearly three times larger than the "
    "calm regime, and cross-asset correlations jump sharply during stress. "
    "On standard out-of-sample backtesting, however, GARCH(1,1) still "
    "outperforms the regime-conditional model because it reacts to volatility "
    "changes one day at a time, while HMM regime detection lags sudden shocks "
    "by days or weeks. Regime models do excel during prolonged stress: in the "
    "2022 rate-hike cycle, regime-conditional CVaR had only 0.4% violations "
    "versus 3-4% for all baselines. The practical takeaway is a two-layer "
    "framework - GARCH for daily risk limits and HMM regime probability as a "
    "strategic overlay for slower-moving macro risks."
)

# ── section content: (sub-heading, body | TABLE_X | FIGURE_X) ───────────────

S_INTRO = [
("", (
    "Most portfolio risk models assume return distributions stay roughly "
    "stable over time. That assumption works fine during normal markets, but "
    "it falls apart during crises - correlations spike, losses cluster, and "
    "distribution tails get much fatter. A model calibrated on mixed calm-"
    "and-crisis data systematically underestimates risk during the exact "
    "periods when accurate estimates matter most."
)),
("", (
    "The metric this paper focuses on is Conditional Value-at-Risk (CVaR), "
    "also called Expected Shortfall - the expected loss given that losses "
    "exceed the 95th percentile. CVaR is preferred over VaR because it "
    "captures how bad losses actually get in the tail, not just where the "
    "tail starts."
)),
("Research Question", (
    "Do regime-aware risk models that combine Bayesian changepoint detection "
    "with dynamic dependence and extreme-value methods produce more reliable "
    "CVaR forecasts than traditional static or rolling-window models?"
)),
("Hypothesis", (
    "Return distributions and cross-asset dependence structures behave "
    "differently across market regimes. Pooling observations from calm and "
    "stress periods leads to systematic underestimation of joint tail risk "
    "during crises. Separating estimation by regime should improve CVaR "
    "forecast accuracy and stability."
)),
("Approach", (
    "I detect regimes using a 2-state Gaussian HMM and a Bayesian changepoint "
    "algorithm (PELT), fit Generalized Pareto Distributions to the loss tails "
    "within each regime, and compare out-of-sample CVaR accuracy against "
    "three baselines: static historical simulation, rolling-window simulation, "
    "and GARCH(1,1)."
)),
]

S_METHOD = [
("2.1  Data", (
    "I downloaded daily adjusted closing prices from Yahoo Finance for ten "
    "equity instruments: SPY (S&P 500) and nine sector SPDR ETFs (XLF, XLE, "
    "XLV, XLK, XLI, XLP, XLU, XLY, XLB), covering January 2005 through "
    "December 2024 - 5,031 trading days total. The sample intentionally spans "
    "the 2008 Global Financial Crisis, the 2020 COVID crash, and the 2022 "
    "Federal Reserve tightening cycle. VIX was also included as a "
    "supplementary stress indicator. Daily log returns were computed and an "
    "equal-weight portfolio constructed. Data were split into a training set "
    "(2005-2017, 3,271 days) and a test set (2018-2024, 1,760 days). Table 1 "
    "and Figure 1 summarize the portfolio's return properties."
)),
("", "TABLE_1"),
("", "FIGURE_1"),
("2.2  Regime Detection", (
    "Two complementary methods were used. The PELT algorithm with a BIC "
    "penalty was applied to rolling 21-day realized volatility, detecting 36 "
    "structural breakpoints - with key dates aligning to September 2008, "
    "February 2020, and May 2022 (Figure 3). A 2-state Gaussian HMM was "
    "then fitted to a feature matrix of rolling volatility, rolling mean "
    "return, and VIX level. The HMM produces both a soft posterior probability "
    "over states and a hard Viterbi assignment. Both states are highly "
    "persistent: P(stay low-vol) = 99.1%, P(stay high-vol) = 98.1% "
    "(Table 2, Figure 2)."
)),
("", "TABLE_2"),
("", "FIGURE_3"),
("", "FIGURE_2"),
("2.3  Tail Risk Modeling", (
    "For each detected regime, a Generalized Pareto Distribution (GPD) was "
    "fitted to loss exceedances above the 85th-percentile threshold using "
    "maximum likelihood. CVaR is computed analytically from the GPD shape "
    "parameter xi and scale parameter beta using the McNeil and Frey (2000) "
    "formula. Three baseline models were also estimated: Static Historical "
    "CVaR (full training sample, constant forecast), Rolling-Window CVaR "
    "(252-day sliding window), and GARCH(1,1) CVaR with normal innovations."
)),
("2.4  Copula Analysis", (
    "To model joint tail behavior across assets, I fitted a Gaussian copula "
    "(zero tail dependence baseline), a Student-t copula with estimated "
    "degrees of freedom, and a bivariate Clayton copula capturing asymmetric "
    "lower-tail dependence between SPY and XLF."
)),
("2.5  Backtesting", (
    "Regime-conditional models were evaluated via an expanding-window "
    "walk-forward backtest with quarterly re-estimation and a 3-year minimum "
    "burn-in. Accuracy was assessed using the Kupiec (1995) Proportion-of-"
    "Failures test, the Christoffersen (1998) Conditional Coverage test, "
    "and pairwise Diebold-Mariano forecast comparison."
)),
]

S_RESULTS = [
("4.1  Regime Characteristics", (
    "The HMM assigns 67.8% of days to the low-vol regime and 32.2% to the "
    "high-vol regime. Table 3 and Figure 4 show the key differences."
)),
("", "TABLE_3"),
("", "FIGURE_4"),
("", (
    "The high-vol regime produces CVaR roughly 2.8x larger (4.42% vs. 1.56%) "
    "and excess kurtosis over four times higher (6.05 vs. 1.39). Annualized "
    "returns are sharply negative (-6.8% vs. +16.5%). Figure 2 shows the "
    "HMM high-vol posterior probability spiking cleanly during GFC 2008, "
    "COVID 2020, and the 2022 rate-hike cycle."
)),
("4.2  Correlation Breakdown in Stress", (
    "Cross-asset correlations rise sharply in the high-vol regime. XLU "
    "(Utilities) is the most extreme: its correlation with SPY nearly doubles "
    "from 0.43 to 0.76. A static model calibrated on pooled data misses this "
    "entirely, understating joint tail risk precisely when it is highest."
)),
("", "TABLE_4"),
("4.3  EVT Results", (
    "Table 5 shows GPD parameters by regime. In the high-vol regime, "
    "xi = +0.116 confirms heavy Pareto-style tails. In the low-vol regime, "
    "xi = -0.073 indicates a bounded thin-tailed distribution. The pooled "
    "full-sample estimate (xi = +0.181) overstates tail heaviness in calm "
    "periods while understating it in stress. Figure 7 validates the GPD "
    "fit via a mean excess plot."
)),
("", "TABLE_5"),
("", "FIGURE_7"),
("4.4  Copula Analysis", (
    "The Student-t copula on all ten assets gives nu = 4.30, indicating "
    "substantial joint tail dependence. The Clayton copula on SPY-XLF gives "
    "lower tail dependence lambda = 0.770 - nearly 3 in 4 extreme SPY losses "
    "coincide with an extreme XLF loss. Tail dependence peaks during COVID "
    "2020 (lambda = 0.694), confirming simultaneous crashes were especially "
    "likely during that episode (Table 6, Figure 5)."
)),
("", "TABLE_6"),
("", "FIGURE_5"),
("4.5  Out-of-Sample Backtesting", (
    "Table 7 shows the full backtest over the 2018-2024 test period. "
    "A properly calibrated 95% CVaR model should produce a 5% violation rate."
)),
("", "TABLE_7"),
("", "FIGURE_6"),
("", (
    "No model hits 5%. Static-Hist is the worst at 1.48% - overstating risk "
    "and tying up unnecessary capital. GARCH comes closest (3.58%) with the "
    "lowest MAE (0.021). Diebold-Mariano tests confirm GARCH beats Regime-EVT "
    "on squared forecast error (DM = 5.71, p < 0.001)."
)),
("4.6  Stress-Period Analysis", (
    "Table 8 and Figure 8 break down violation rates during the named stress "
    "events."
)),
("", "TABLE_8"),
("", "FIGURE_8"),
("", (
    "The split is stark. During COVID 2020 regime models perform worst "
    "(28.9% violations) - the HMM was still in low-vol mode when the crash "
    "started. During the 2022 rate-hike cycle, regime models are best by "
    "far (0.4% violations vs. 3-4% for baselines). Once the HMM locked onto "
    "the high-vol state, it held it through the entire cycle."
)),
]

S_DISCUSSION = [
("5.1  What the Results Mean", (
    "The hypothesis gets partial support. Regime conditioning clearly captures "
    "something real - tail distributions, correlations, and joint tail "
    "dependence all differ significantly across regimes. Any risk framework "
    "that ignores this structure is missing important information. That said, "
    "the regime detection lag is the core problem: by the time the HMM is "
    "confident the regime has shifted, the worst losses may already be behind "
    "you. GARCH avoids this because it responds to yesterday's realized "
    "return directly, adapting at a one-day lag instead of a multi-week lag."
)),
("5.2  Implications for the Fund", (
    "The most practical takeaway is a two-layer framework. Use GARCH for "
    "daily CVaR limits and position sizing - it is the best single tool for "
    "day-to-day risk management. Layer HMM regime probability on top as a "
    "strategic overlay: when P(high-vol) crosses ~40%, scale back exposure "
    "pre-emptively before volatility fully materializes. The 2022 result "
    "shows regime models are very effective once a sustained stress "
    "environment is established. The copula analysis also informs hedging - "
    "when tail dependence spikes, apparent diversifiers stop providing "
    "protection."
)),
("5.3  Limitations", (
    "The regime detection lag is the main limitation. A forward-looking "
    "indicator - credit spreads or VIX term structure slope - could help "
    "reduce it. The equal-weight portfolio simplifies the analysis; a real "
    "fund portfolio with time-varying weights would require CVaR computed "
    "on the actual weight vector. Copula-based CVaR was estimated in-sample "
    "only, and PELT was applied to the full sample for visualization - a "
    "live implementation would need BOCPD."
)),
]

S_CONCLUSION = [
("", (
    "Market regimes genuinely matter for tail risk. The high-vol regime has "
    "fundamentally different tail properties, higher correlations, and greater "
    "joint crash risk - mixing the two in one model distorts estimates in "
    "both directions. But the regime detection lag prevents this from "
    "translating into better out-of-sample CVaR numbers during sudden shocks "
    "like COVID 2020. GARCH handles those better."
)),
("", (
    "The regime approach is not a replacement for GARCH - it is a complement. "
    "As a strategic overlay, it flags elevated tail dependence before a "
    "sustained stress cycle fully materializes and delivers significantly "
    "better CVaR estimates during slow-moving macro stress like 2022. The "
    "recommended approach for the fund is to run both in parallel - GARCH "
    "for daily limits, HMM regime probability for strategic capital "
    "allocation decisions."
)),
("Future Work", (
    "The most important extension is incorporating forward-looking signals "
    "(credit spreads, VIX term structure) to reduce the regime detection "
    "lag. Fitting regime-specific copulas and running a proper walk-forward "
    "copula backtest would also strengthen the multivariate tail analysis."
)),
]

REFERENCES_LIST = [
    ("Christoffersen, P. F. (1998).",
     "Evaluating interval forecasts. International Economic Review, 39(4), 841-862."),
    ("Diebold, F. X., & Mariano, R. S. (1995).",
     "Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253-263."),
    ("Embrechts, P., Kluppelberg, C., & Mikosch, T. (1997).",
     "Modelling extremal events for insurance and finance. Springer."),
    ("Engle, R. F. (1982).",
     "Autoregressive conditional heteroscedasticity with estimates of the variance "
     "of United Kingdom inflation. Econometrica, 50(4), 987-1007."),
    ("Hamilton, J. D. (1989).",
     "A new approach to the economic analysis of nonstationary time series and the "
     "business cycle. Econometrica, 57(2), 357-384."),
    ("Kupiec, P. H. (1995).",
     "Techniques for verifying the accuracy of risk measurement models. "
     "Journal of Derivatives, 3(2), 73-84."),
    ("McNeil, A. J., & Frey, R. (2000).",
     "Estimation of tail-related risk measures for heteroscedastic financial time "
     "series: An extreme value approach. Journal of Empirical Finance, 7(3-4), 271-300."),
    ("McNeil, A. J., Frey, R., & Embrechts, P. (2005).",
     "Quantitative risk management: Concepts, techniques and tools. "
     "Princeton University Press."),
    ("Nelsen, R. B. (2006).",
     "An introduction to copulas (2nd ed.). Springer."),
    ("Rockafellar, R. T., & Uryasev, S. (2000).",
     "Optimization of conditional value-at-risk. Journal of Risk, 2(3), 21-41."),
]

DISCLAIMER_TEXT = (
    "The information, trading strategies, and materials presented in this "
    "report are provided strictly for educational and informational purposes "
    "and are offered without any guarantees or warranties regarding their "
    "accuracy, completeness, reliability, or timeliness. These materials are "
    "not intended to serve as financial, legal, tax, investment, or other "
    "professional advice, and no content herein should be interpreted as a "
    "recommendation to buy, sell, or hold any security, financial product, or "
    "instrument, nor as an endorsement of any specific strategy, practice, or "
    "course of action. Users are strongly encouraged to conduct their own "
    "independent research and due diligence or seek advice from qualified "
    "professionals before making any financial or investment decisions. "
    "Trading and investing involve substantial risks, including potential loss "
    "of principal. Past performance is not indicative of future results. By "
    "accessing this material, users accept all responsibility for risks "
    "inherent in any activity based on information presented herein. "
    "The creators disclaim all liability for any losses arising from reliance "
    "on this material to the fullest extent permitted by applicable law."
)

# ── tables (widths must sum <= USABLE_W = 165.1 mm) ─────────────────────────

TABLES = {
"TABLE_1": {
    "title": "Table 1: Portfolio return summary statistics (2005-2024)",
    "headers": ["Statistic", "Value"],
    "rows": [
        ["Mean daily return",   "0.0356%"],
        ["Std deviation",       "1.169%"],
        ["Minimum (worst day)", "-12.25%"],
        ["Maximum (best day)",  "+10.62%"],
        ["Excess kurtosis",     "14.14  (Normal = 0)"],
        ["Skewness",            "-0.59"],
    ],
    "widths": [95, 68],
},
"TABLE_2": {
    "title": "Table 2: HMM estimated transition matrix",
    "headers": ["", "To: Low-vol", "To: High-vol"],
    "rows": [
        ["From: Low-vol",  "0.9911", "0.0089"],
        ["From: High-vol", "0.0187", "0.9813"],
    ],
    "widths": [57, 52, 52],
},
"TABLE_3": {
    "title": "Table 3: Regime characteristics - HMM 2-state model (2005-2024)",
    "headers": ["Statistic", "Low-vol Regime", "High-vol Regime"],
    "rows": [
        ["Days (% of sample)", "3,396  (67.8%)", "1,615  (32.2%)"],
        ["Ann. mean return",   "+16.54%",         "-6.81%"],
        ["Ann. volatility",    "10.76%",           "28.77%"],
        ["Excess kurtosis",    "1.39",             "6.05"],
        ["Skewness",           "-0.345",           "-0.369"],
        ["CVaR (95%)",         "1.56%",            "4.42%"],
    ],
    "widths": [65, 48, 48],
},
"TABLE_4": {
    "title": "Table 4: Cross-asset return correlation by regime (selected pairs)",
    "headers": ["Asset Pair", "Low-vol", "High-vol", "Change"],
    "rows": [
        ["SPY - XLF", "0.815", "0.857", "+0.042"],
        ["SPY - XLE", "0.580", "0.790", "+0.210"],
        ["SPY - XLU", "0.428", "0.756", "+0.328"],
        ["SPY - XLV", "0.741", "0.859", "+0.118"],
        ["SPY - XLK", "0.869", "0.934", "+0.065"],
    ],
    "widths": [58, 34, 34, 34],
},
"TABLE_5": {
    "title": "Table 5: GPD parameters by regime (training set 2005-2017)",
    "headers": ["Regime", "n", "xi (shape)", "beta (scale)", "CVaR 95%"],
    "rows": [
        ["Full sample", "3,271", " 0.181", "0.00847", "2.92%"],
        ["Low-vol",     "2,219", "-0.073", "0.00537", "1.56%"],
        ["High-vol",    "1,052", "+0.116", "0.01237", "4.47%"],
    ],
    "widths": [42, 24, 34, 37, 28],
},
"TABLE_6": {
    "title": "Table 6: Student-t copula tail dependence by stress period",
    "headers": ["Period", "Est. nu", "Bivariate lambda"],
    "rows": [
        ["Full sample (2005-2024)", "4.30",  "0.543"],
        ["GFC 2008-09",            "7.98",  "0.458"],
        ["COVID 2020",             "6.69",  "0.694"],
        ["Rate Hike 2022",         "10.85", "0.429"],
    ],
    "widths": [78, 36, 48],
},
"TABLE_7": {
    "title": "Table 7: CVaR model comparison - out-of-sample test period 2018-2024",
    "headers": ["Model", "Viol Rate", "Expected", "Kupiec p", "CC p", "MAE"],
    "rows": [
        ["Static-Hist", "1.48%", "5.0%", "<0.001*", "<0.001*", "0.0300"],
        ["Rolling-252", "2.67%", "5.0%", "<0.001*", "<0.001*", "0.0274"],
        ["GARCH",       "3.58%", "5.0%", "0.004*",  "0.009*",  "0.0205"],
        ["Regime-Hist", "3.30%", "5.0%", "<0.001*", "<0.001*", "0.0280"],
        ["Regime-EVT",  "3.24%", "5.0%", "<0.001*", "<0.001*", "0.0282"],
    ],
    "widths": [38, 25, 25, 27, 27, 25],
    "note": "* Rejects H0 (violation rate = 5%) at 5% significance.",
},
"TABLE_8": {
    "title": "Table 8: CVaR violation rates during named stress periods",
    "headers": ["Model", "Full Sample", "COVID 2020", "Rate Hike 2022"],
    "rows": [
        ["Static-Hist", "1.48%", "25.00%", "3.19%"],
        ["Rolling-252", "2.67%", "17.31%", "4.38%"],
        ["GARCH",       "3.58%", "11.54%", "3.59%"],
        ["Regime-Hist", "3.30%", "28.85%", "0.40%"],
        ["Regime-EVT",  "3.24%", "28.85%", "0.40%"],
    ],
    "widths": [42, 40, 40, 42],
},
}

FIGURES = {
    "FIGURE_1": ("fig_eda.png",               162, "Figure 1: Portfolio returns, rolling volatility, and VIX (2005-2024). Shaded bands mark GFC 2008-09, COVID 2020, and Rate Hike 2022."),
    "FIGURE_2": ("fig_hmm_states.png",        162, "Figure 2: HMM posterior state probabilities. Green = P(low-vol), red = P(high-vol)."),
    "FIGURE_3": ("fig_bcp_regimes.png",       162, "Figure 3: PELT changepoint segmentation with 36 detected structural breaks (red dashed lines)."),
    "FIGURE_4": ("fig_loss_distributions.png",162, "Figure 4: Loss distributions by HMM regime (left) and QQ-plot vs. Normal (right)."),
    "FIGURE_5": ("fig_tail_dependence.png",   128, "Figure 5: Pairwise tail dependence heatmap - Student-t copula (nu = 4.30)."),
    "FIGURE_6": ("fig_cvar_comparison.png",   162, "Figure 6: CVaR forecast time series for all five models, 2018-2024 test period."),
    "FIGURE_7": ("fig_mean_excess.png",       120, "Figure 7: Mean excess plot confirming GPD fit validity (upward linear trend implies xi > 0)."),
    "FIGURE_8": ("fig_violations.png",        162, "Figure 8: CVaR violation days (red dots) per model, 2018-2024 test period."),
}


# ── PDF class ────────────────────────────────────────────────────────────────

class AlgoGatorsPDF(FPDF):

    def header(self):
        # Skip header on title page (page 1)
        if self.page_no() == 1:
            return
        # Logo left-aligned
        if os.path.exists(LOGO_FILE):
            self.image(LOGO_FILE, x=self.l_margin, y=6, w=LOGO_W)
        # Thin divider line
        self.set_y(14)
        self.set_draw_color(*LGRAY)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(),
                  self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        # No footer on title page
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        # Page number starts at 2 (matching template footer)
        self.cell(0, 8, str(self.page_no()), align="C")


# ── helpers ──────────────────────────────────────────────────────────────────

def _scale_widths(widths):
    """Scale table column widths down if they exceed usable page width."""
    total = sum(widths)
    if total > USABLE_W:
        f = USABLE_W / total
        return [w * f for w in widths]
    return list(widths)


def section_heading(pdf, num, text):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*HEAD_COL)
    pdf.cell(0, 9, f"{num}  {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*HEAD_COL)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    pdf.set_text_color(*BLACK)


def sub_heading(pdf, text):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*HEAD_COL)
    pdf.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    pdf.set_text_color(*BLACK)


def body(pdf, text):
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def draw_table(pdf, key):
    tbl    = TABLES[key]
    widths = _scale_widths(tbl["widths"])

    pdf.ln(2)
    pdf.set_font("Helvetica", "BI", 9)
    pdf.set_text_color(*HEAD_COL)
    pdf.multi_cell(0, 5, tbl["title"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    # Header row
    pdf.set_fill_color(*TBL_HDR)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 8.5)
    for i, h in enumerate(tbl["headers"]):
        pdf.cell(widths[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 8.5)
    for ri, row in enumerate(tbl["rows"]):
        pdf.set_fill_color(*ROW_A) if ri % 2 == 0 else pdf.set_fill_color(*ROW_B)
        pdf.set_text_color(*BLACK)
        for i, cell in enumerate(row):
            align = "L" if i == 0 else "C"
            pdf.cell(widths[i], 6, cell, border=1, fill=True, align=align)
        pdf.ln()

    if "note" in tbl:
        pdf.set_font("Helvetica", "I", 7.5)
        pdf.set_text_color(*GRAY)
        pdf.cell(0, 5, tbl["note"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(3)
    pdf.set_text_color(*BLACK)


def draw_figure(pdf, key):
    fname, max_w, caption = FIGURES[key]
    if not os.path.exists(fname):
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(*GRAY)
        pdf.cell(0, 6, f"[{fname} not found]",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        return

    with PILImage.open(fname) as im:
        pw, ph = im.size
    render_w = min(float(max_w), USABLE_W)
    render_h = render_w * ph / pw

    # Add new page if image + caption won't fit
    bottom_margin = 20
    if pdf.get_y() + render_h + 12 > PAGE_H - bottom_margin:
        pdf.add_page()

    pdf.ln(2)
    x_off = pdf.l_margin + (USABLE_W - render_w) / 2.0
    pdf.image(fname, x=x_off, w=render_w)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(*GRAY)
    pdf.ln(1)
    pdf.multi_cell(0, 5, caption, align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    pdf.set_text_color(*BLACK)


def render_blocks(pdf, blocks):
    for label, content in blocks:
        if label:
            sub_heading(pdf, label)
        if isinstance(content, str) and content.startswith("TABLE_"):
            draw_table(pdf, content)
        elif isinstance(content, str) and content.startswith("FIGURE_"):
            draw_figure(pdf, content)
        else:
            body(pdf, content)


# ── individual pages ─────────────────────────────────────────────────────────

def title_page(pdf):
    pdf.add_page()

    # AlgoGators logo centered at top
    if os.path.exists(LOGO_FILE):
        logo_render_w = 70.0
        x_logo = (PAGE_W - logo_render_w) / 2
        pdf.image(LOGO_FILE, x=x_logo, y=22, w=logo_render_w)

    pdf.set_y(52)
    pdf.set_draw_color(*HEAD_COL)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(8)

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*HEAD_COL)
    pdf.multi_cell(0, 9, TITLE, align="C")
    pdf.ln(5)

    # Author / org / date
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*BLACK)
    for line in [AUTHOR, ORG, DATE]:
        pdf.cell(0, 7, line, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # GitHub link
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 6, f"Code: {GITHUB_URL}", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(3)
    pdf.set_draw_color(*HEAD_COL)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())

    # ── Section 1: Abstract (on title page, matching template) ───────────────
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*HEAD_COL)
    pdf.cell(0, 8, "1  Abstract", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(0, 6, ABSTRACT_TEXT, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def references_page(pdf):
    pdf.add_page()
    section_heading(pdf, 7, "References")
    for authors, text in REFERENCES_LIST:
        pdf.set_font("Helvetica", "B", 10.5)
        pdf.set_text_color(*BLACK)
        pdf.multi_cell(0, 6, authors, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.multi_cell(0, 6, "    " + text,
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)


def appendices_page(pdf):
    pdf.add_page()
    section_heading(pdf, 8, "Appendices")
    body(pdf,
         "All figures are embedded inline within their respective sections. "
         "Source code, data pipeline, and model implementations are available "
         f"at: {GITHUB_URL}")

    # Figure index table
    pdf.set_font("Helvetica", "BI", 9)
    pdf.set_text_color(*HEAD_COL)
    pdf.cell(0, 5, "Figure Index", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    col_w = [28, 52, 82]
    pdf.set_fill_color(*TBL_HDR)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 8.5)
    for h, w in zip(["Figure", "File", "Description"], col_w):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()

    items = [
        ("Figure 1", "fig_eda.png",               "Returns, rolling vol, VIX with stress shading"),
        ("Figure 2", "fig_hmm_states.png",         "HMM posterior state probabilities"),
        ("Figure 3", "fig_bcp_regimes.png",        "PELT changepoint segmentation"),
        ("Figure 4", "fig_loss_distributions.png", "Loss distributions by regime + QQ-plot"),
        ("Figure 5", "fig_tail_dependence.png",    "Pairwise tail dependence heatmap"),
        ("Figure 6", "fig_cvar_comparison.png",    "CVaR forecast comparison, 2018-2024"),
        ("Figure 7", "fig_mean_excess.png",        "Mean excess plot (GPD validation)"),
        ("Figure 8", "fig_violations.png",         "CVaR violations by model, 2018-2024"),
    ]
    pdf.set_font("Helvetica", "", 8.5)
    for ri, (fig, fname, desc) in enumerate(items):
        pdf.set_fill_color(*ROW_A) if ri % 2 == 0 else pdf.set_fill_color(*ROW_B)
        pdf.set_text_color(*BLACK)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(col_w[0], 6, fig,   border=1, fill=True, align="C")
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(col_w[1], 6, fname, border=1, fill=True, align="C")
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(col_w[2], 6, desc,  border=1, fill=True, align="L")
        pdf.ln()


def disclaimer_page(pdf):
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*HEAD_COL)
    pdf.cell(0, 7, "Disclaimer", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*HEAD_COL)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(0, 5.2, DISCLAIMER_TEXT,
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# ── build ─────────────────────────────────────────────────────────────────────

def build():
    pdf = AlgoGatorsPDF(orientation="P", unit="mm", format="Letter")
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    # top margin larger to leave room for header logo on body pages
    pdf.set_auto_page_break(auto=True, margin=18)

    title_page(pdf)

    pdf.add_page(); section_heading(pdf, 2, "Introduction");  render_blocks(pdf, S_INTRO)
    pdf.add_page(); section_heading(pdf, 3, "Methodology");   render_blocks(pdf, S_METHOD)
    pdf.add_page(); section_heading(pdf, 4, "Results");       render_blocks(pdf, S_RESULTS)
    pdf.add_page(); section_heading(pdf, 5, "Discussion");    render_blocks(pdf, S_DISCUSSION)
    pdf.add_page(); section_heading(pdf, 6, "Conclusion");    render_blocks(pdf, S_CONCLUSION)
    references_page(pdf)
    appendices_page(pdf)
    disclaimer_page(pdf)

    pdf.output(OUT)
    print(f"Saved: {OUT}  ({pdf.page} pages)")


if __name__ == "__main__":
    build()
