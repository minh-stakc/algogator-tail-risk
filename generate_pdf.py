"""
generate_pdf.py
Generates the AlgoGators rough draft PDF matching the official template.
Template sections: 1-Abstract  2-Introduction  3-Methodology  4-Results
                   5-Discussion  6-Conclusion  7-References  8-Appendices
                   + Disclaimer
"""

import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image as PILImage

TITLE  = "Regime-Conditional Tail Risk Modeling for Portfolio Risk Management"
AUTHOR = "Nathan Hoang"
ORG    = "AlgoGators Capstone Project"
DATE   = "March 2026"
OUT    = "rough_draft.pdf"

# Template colour palette (from theme1.xml)
DARK_BLUE = (68,  84, 106)   # #44546A  - headings
ACCENT    = (68, 114, 196)   # #4472C4  - accent / table header
BLACK     = (30,  30,  30)
GRAY      = (110, 110, 110)
LGRAY     = (220, 220, 220)
WHITE     = (255, 255, 255)
ROW_EVEN  = (235, 241, 251)
ROW_ODD   = (255, 255, 255)

MARGIN   = 25.4   # 1-inch margins (standard Word default)
PAGE_W   = 215.9
USABLE_W = PAGE_W - 2 * MARGIN   # 165.1 mm

# ── content ─────────────────────────────────────────────────────────────────

ABSTRACT_TEXT = (
    "This paper investigates whether regime-aware risk models produce better "
    "Conditional Value-at-Risk (CVaR) forecasts than standard portfolio risk "
    "approaches. Using daily returns from the S&P 500 and nine sector ETFs "
    "over 2005-2024, I detect market regimes with a Hidden Markov Model (HMM) "
    "and estimate tail risk within each regime using Peaks-over-Threshold "
    "extreme value theory. The results are mixed. Regime conditioning reveals "
    "something economically real - the high-volatility regime has a CVaR "
    "nearly three times larger than the calm regime, and cross-asset "
    "correlations jump sharply during stress periods. On standard "
    "out-of-sample backtesting, however, GARCH(1,1) still outperforms the "
    "regime-conditional model because it reacts to volatility changes one "
    "day at a time, while HMM regime detection can lag sudden shocks by "
    "days or weeks. Regime models do excel during prolonged stress: in the "
    "2022 rate-hike cycle, regime-conditional CVaR had only 0.4% violations "
    "versus 3-4% for all baselines. The practical takeaway is a two-layer "
    "framework - GARCH for daily risk limits and HMM regime probability as "
    "a strategic overlay for slower-moving macro risks."
)

# Each block: (sub-heading or "", body_text | "TABLE_X" | "FIGURE_X")
INTRO_BLOCKS = [
("", (
    "Most portfolio risk models assume that return distributions stay roughly "
    "stable over time. That assumption works fine during normal markets, but "
    "it falls apart during crises - when correlations spike, losses cluster, "
    "and the tails of the return distribution get much fatter. A model "
    "calibrated on mixed calm-and-crisis data will systematically "
    "underestimate risk during the exact periods when accurate estimates "
    "matter most."
)),
("", (
    "The metric this paper focuses on is Conditional Value-at-Risk (CVaR), "
    "also called Expected Shortfall - the expected loss given that losses "
    "exceed the 95th percentile on any given day. CVaR is preferred over "
    "plain VaR because it captures how bad losses actually get in the tail, "
    "not just where the tail starts."
)),
("Research Question", (
    "Do regime-aware risk models that combine Bayesian changepoint detection "
    "with dynamic dependence and extreme-value methods produce more reliable "
    "CVaR forecasts than traditional static or rolling-window models?"
)),
("Hypothesis", (
    "Return distributions and cross-asset dependence structures behave "
    "differently across market regimes. Pooling observations from calm and "
    "stress periods into one model hides this variation and leads to "
    "systematic underestimation of joint tail risk during crises. Separating "
    "estimation by regime should improve CVaR forecast accuracy and stability."
)),
("Approach", (
    "I detect regimes using a 2-state Gaussian HMM and a Bayesian changepoint "
    "algorithm (PELT), fit Generalized Pareto Distributions to the loss tails "
    "within each regime, and compare out-of-sample CVaR accuracy against three "
    "baselines: static historical simulation, rolling-window historical "
    "simulation, and GARCH(1,1)."
)),
]

METHOD_BLOCKS = [
("2.1  Data", (
    "I downloaded daily adjusted closing prices from Yahoo Finance for ten "
    "equity instruments: SPY (S&P 500) and nine sector SPDR ETFs (XLF, XLE, "
    "XLV, XLK, XLI, XLP, XLU, XLY, XLB), covering January 2005 through "
    "December 2024 - 5,031 trading days total. The sample intentionally spans "
    "the 2008 Global Financial Crisis, the 2020 COVID crash, and the 2022 "
    "Federal Reserve tightening cycle. VIX was also downloaded as a "
    "supplementary stress indicator. Daily log returns were computed and an "
    "equal-weight portfolio constructed. Data were split into a training set "
    "(2005-2017, 3,271 days) and a test set (2018-2024, 1,760 days). Table 1 "
    "and Figure 1 summarize the portfolio's return properties."
)),
("", "TABLE_1"),
("", "FIGURE_1"),
("2.2  Regime Detection", (
    "I used two complementary methods. The PELT algorithm with a BIC penalty "
    "was applied to rolling 21-day realized volatility, detecting 36 "
    "structural breakpoints across the sample - with key dates aligning to "
    "September 2008, February 2020, and May 2022 (Figure 3). A 2-state "
    "Gaussian HMM was then fitted to a feature matrix of rolling volatility, "
    "rolling mean return, and VIX level. The HMM produces both a soft "
    "posterior probability over states and a hard Viterbi assignment. "
    "Both states are highly persistent: P(stay low-vol) = 99.1%, "
    "P(stay high-vol) = 98.1% (Table 2, Figure 2)."
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
    "degrees of freedom, and a bivariate Clayton copula capturing lower-tail "
    "dependence between SPY and XLF."
)),
("2.5  Backtesting", (
    "Regime-conditional models were evaluated via an expanding-window "
    "walk-forward backtest with quarterly re-estimation and a 3-year minimum "
    "burn-in. Accuracy was assessed using the Kupiec (1995) "
    "Proportion-of-Failures test, the Christoffersen (1998) Conditional "
    "Coverage test, and pairwise Diebold-Mariano forecast comparison."
)),
]

RESULTS_BLOCKS = [
("4.1  Regime Characteristics", (
    "The HMM assigns 67.8% of days to the low-vol regime and 32.2% to the "
    "high-vol regime. The economic difference between the two is large."
)),
("", "TABLE_3"),
("", "FIGURE_4"),
("", (
    "The high-vol regime produces CVaR roughly 2.8x larger than the low-vol "
    "regime (4.42% vs. 1.56%), and excess kurtosis over four times higher "
    "(6.05 vs. 1.39). Figure 2 shows how the HMM posterior probability of "
    "the high-vol state spikes cleanly during GFC 2008, COVID 2020, and the "
    "2022 rate-hike cycle."
)),
("4.2  Correlation Breakdown", (
    "Cross-asset correlations rise sharply in the high-vol regime. The "
    "Utilities sector (XLU) is the most extreme example - its correlation "
    "with SPY nearly doubles from 0.43 to 0.76. A static model calibrated "
    "on pooled data misses this completely, understating joint tail risk "
    "during the exact periods when it is highest."
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
    "substantial joint tail dependence. The Clayton copula on SPY-XLF "
    "gives lower tail dependence lambda = 0.770 - nearly 3 in 4 extreme "
    "SPY losses coincide with an extreme XLF loss. Tail dependence peaks "
    "during COVID 2020 (lambda = 0.694), confirming that simultaneous "
    "crashes were especially likely during that episode (Table 6, Figure 5)."
)),
("", "TABLE_6"),
("", "FIGURE_5"),
("4.5  Out-of-Sample Backtesting", (
    "Table 7 shows the full backtest over the 2018-2024 test period. "
    "A properly calibrated 95% CVaR model should have a 5% violation rate."
)),
("", "TABLE_7"),
("", "FIGURE_6"),
("", (
    "No model hits 5%. Static-Hist is the worst at 1.48% - way too "
    "conservative, tying up unnecessary capital. GARCH comes closest (3.58%) "
    "with the lowest MAE (0.021). Diebold-Mariano tests confirm GARCH beats "
    "Regime-EVT on squared forecast error (DM = 5.71, p < 0.001)."
)),
("4.6  Stress-Period Analysis", "TABLE_8"),
("", "FIGURE_8"),
("", (
    "The split is stark. During COVID 2020, regime models perform worst "
    "(28.9% violations) - the HMM was still in low-vol mode when the crash "
    "started. During the 2022 rate-hike cycle, regime models are best by "
    "far (0.4% violations vs. 3-4% for all baselines). Once the HMM "
    "locked onto the high-vol state, it held it through the entire cycle."
)),
]

DISCUSSION_BLOCKS = [
("5.1  What the Results Mean", (
    "The hypothesis gets partial support. Regime conditioning clearly "
    "captures something real - tail distributions, correlations, and joint "
    "tail dependence all differ significantly across regimes. Any risk "
    "framework that ignores this structure is missing important information. "
    "That said, turning that insight into better real-time CVaR forecasts is "
    "harder than expected. The regime detection lag is the core problem: by "
    "the time the HMM is confident the regime has shifted, the worst losses "
    "may already be behind you. GARCH avoids this because it responds to "
    "yesterday's realized return directly, adapting at a one-day lag instead "
    "of multi-week lag."
)),
("5.2  Implications for the Fund", (
    "The most practical takeaway is a two-layer framework. Use GARCH for "
    "daily CVaR limits and position sizing - it is the best single tool for "
    "day-to-day risk management. Use HMM regime probability as a strategic "
    "overlay: when P(high-vol) crosses ~40%, scale back exposure "
    "pre-emptively before volatility fully materializes. The 2022 result "
    "shows regime models are very effective once a sustained stress "
    "environment is established. The copula analysis also provides useful "
    "input for hedging - when tail dependence spikes, apparent diversifiers "
    "stop providing protection."
)),
("5.3  Limitations", (
    "The regime detection lag is the main limitation. A forward-looking "
    "indicator - credit spreads, VIX term structure slope - could help "
    "reduce it. The equal-weight portfolio simplifies the analysis; a "
    "real portfolio with time-varying weights would require CVaR "
    "computed on the actual weight vector. Copula-based CVaR was estimated "
    "in-sample only, and PELT was applied to the full sample - a live "
    "implementation would need an online variant such as BOCPD."
)),
]

CONCLUSION_BLOCKS = [
("", (
    "Market regimes genuinely matter for tail risk. The high-vol regime "
    "has fundamentally different tail properties, higher correlations, and "
    "greater joint crash risk - and mixing the two in one model distorts "
    "estimates in both directions. But the regime detection lag prevents "
    "this from translating into better out-of-sample CVaR numbers during "
    "sudden shocks like COVID 2020. GARCH handles those better."
)),
("", (
    "The regime approach is not a replacement for GARCH - it is a "
    "complement. As a strategic overlay, it adds genuine value: it flags "
    "elevated tail dependence before a sustained stress cycle fully "
    "materializes and delivers significantly better CVaR estimates during "
    "slow-moving macro stress like 2022. For the fund, the recommended "
    "approach is to run both in parallel - GARCH for daily limits, "
    "HMM regime probability for strategic capital allocation."
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

DISCLAIMER = (
    "The information, trading strategies, and materials presented in this report "
    "are provided strictly for educational and informational purposes and are "
    "offered without any guarantees or warranties regarding their accuracy, "
    "completeness, reliability, or timeliness. These materials are not intended "
    "to serve as financial, legal, tax, investment, or other professional advice, "
    "and no content herein should be interpreted as a recommendation to buy, sell, "
    "or hold any security, financial product, or instrument, nor as an endorsement "
    "of any specific strategy, practice, or course of action. Users of this "
    "material are strongly encouraged to conduct their own independent research, "
    "analysis, and due diligence or seek advice from qualified professionals before "
    "making any financial or investment decisions. The authors, contributors, and "
    "distributors of this material are not acting as fiduciaries and do not assume "
    "any legal or ethical duty to the user. Trading and investing involve "
    "substantial risks, including potential loss of principal. Past performance is "
    "not indicative of future results. By accessing this material, users accept all "
    "responsibility for risks inherent in any trading or investment activity based "
    "on information presented herein."
)

# ── tables ───────────────────────────────────────────────────────────────────

TABLES = {
"TABLE_1": {
    "title": "Table 1: Portfolio return summary statistics (2005-2024)",
    "headers": ["Statistic", "Value"],
    "rows": [
        ["Mean daily return",  "0.0356%"],
        ["Std deviation",      "1.169%"],
        ["Minimum (worst day)","-12.25%"],
        ["Maximum (best day)", "+10.62%"],
        ["Excess kurtosis",    "14.14  (Normal = 0)"],
        ["Skewness",           "-0.59"],
    ],
    "widths": [95, 70],
},
"TABLE_2": {
    "title": "Table 2: HMM transition matrix",
    "headers": ["", "To: Low-vol", "To: High-vol"],
    "rows": [
        ["From: Low-vol",  "0.9911", "0.0089"],
        ["From: High-vol", "0.0187", "0.9813"],
    ],
    "widths": [60, 52, 52],
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
    "widths": [65, 50, 50],
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
    "widths": [60, 35, 35, 35],
},
"TABLE_5": {
    "title": "Table 5: GPD parameters by regime (training set 2005-2017)",
    "headers": ["Regime", "n", "xi (shape)", "beta (scale)", "CVaR 95%"],
    "rows": [
        ["Full sample", "3,271", " 0.181", "0.00847", "2.92%"],
        ["Low-vol",     "2,219", "-0.073", "0.00537", "1.56%"],
        ["High-vol",    "1,052", "+0.116", "0.01237", "4.47%"],
    ],
    "widths": [42, 25, 35, 38, 30],
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
    "widths": [78, 37, 55],
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
    "widths": [38, 26, 26, 26, 26, 26],
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
    "widths": [43, 40, 40, 45],
},
}

FIGURES = {
    "FIGURE_1": ("fig_eda.png",              165, "Figure 1: Portfolio returns, rolling volatility, and VIX (2005-2024). Shaded bands mark GFC 2008-09, COVID 2020, and Rate Hike 2022."),
    "FIGURE_2": ("fig_hmm_states.png",       165, "Figure 2: HMM posterior state probabilities. Green = P(low-vol), red = P(high-vol)."),
    "FIGURE_3": ("fig_bcp_regimes.png",      165, "Figure 3: Bayesian changepoint (PELT) segmentation. Red dashed lines = 36 detected structural breaks."),
    "FIGURE_4": ("fig_loss_distributions.png",165,"Figure 4: Loss distributions by HMM regime (left) and QQ-plot vs. Normal (right)."),
    "FIGURE_5": ("fig_tail_dependence.png",  130, "Figure 5: Pairwise tail dependence heatmap - Student-t copula (nu = 4.30)."),
    "FIGURE_6": ("fig_cvar_comparison.png",  165, "Figure 6: CVaR forecast time series for all five models, 2018-2024 test period."),
    "FIGURE_7": ("fig_mean_excess.png",      125, "Figure 7: Mean excess plot. Upward linear trend validates GPD fit with xi > 0."),
    "FIGURE_8": ("fig_violations.png",       165, "Figure 8: CVaR violation days (red dots) by model, 2018-2024 test period."),
}


# ── PDF class ────────────────────────────────────────────────────────────────

class Paper(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 7, TITLE[:72] + ("..." if len(TITLE) > 72 else ""),
                  align="L", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*LGRAY)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 8, f"Page {self.page_no() - 1}", align="C")


# ── helpers ──────────────────────────────────────────────────────────────────

def section_heading(pdf, num, text):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*DARK_BLUE)
    pdf.cell(0, 8, f"{num}  {text}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*DARK_BLUE)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    pdf.set_text_color(*BLACK)


def sub_heading(pdf, text):
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(*DARK_BLUE)
    pdf.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)
    pdf.set_text_color(*BLACK)


def body(pdf, text):
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(0, 5.8, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)


def draw_table(pdf, key):
    tbl = TABLES[key]
    widths = tbl["widths"]
    # safety: scale down if total > USABLE_W
    total = sum(widths)
    if total > USABLE_W:
        scale = USABLE_W / total
        widths = [w * scale for w in widths]

    pdf.ln(2)
    pdf.set_font("Helvetica", "BI", 9)
    pdf.set_text_color(*DARK_BLUE)
    pdf.multi_cell(0, 5, tbl["title"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    # header row
    pdf.set_fill_color(*ACCENT)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 8.5)
    for i, h in enumerate(tbl["headers"]):
        pdf.cell(widths[i], 7, h, border=1, fill=True, align="C")
    pdf.ln()

    # data rows
    pdf.set_font("Helvetica", "", 8.5)
    for ri, row in enumerate(tbl["rows"]):
        pdf.set_fill_color(*ROW_EVEN) if ri % 2 == 0 else pdf.set_fill_color(*ROW_ODD)
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
        pdf.cell(0, 6, f"[{fname} not found]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        return
    with PILImage.open(fname) as im:
        pw, ph = im.size
    render_w = min(max_w, USABLE_W)
    render_h = render_w * ph / pw
    if pdf.get_y() + render_h + 12 > (279.4 - 20):
        pdf.add_page()
    pdf.ln(2)
    x_off = pdf.l_margin + (USABLE_W - render_w) / 2
    pdf.image(fname, x=x_off, w=render_w)
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(*GRAY)
    pdf.ln(1)
    pdf.multi_cell(0, 5, caption, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
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


# ── pages ────────────────────────────────────────────────────────────────────

def title_page(pdf, github_url=""):
    pdf.add_page()
    pdf.set_fill_color(*DARK_BLUE)
    pdf.rect(0, 0, pdf.w, 68, "F")
    pdf.set_y(14)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*WHITE)
    pdf.multi_cell(0, 9, TITLE, align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    for line in [AUTHOR, ORG, DATE]:
        pdf.cell(0, 7, line, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if github_url:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(150, 200, 255)
        pdf.cell(0, 6, f"Code & data: {github_url}", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Abstract (section 1 on title page)
    pdf.set_y(80)
    pdf.set_text_color(*DARK_BLUE)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "1  Abstract", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*DARK_BLUE)
    pdf.set_line_width(0.5)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    pdf.set_fill_color(242, 245, 251)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.4)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(0, 5.8, ABSTRACT_TEXT, border=1, fill=True,
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def references_page(pdf):
    pdf.add_page()
    section_heading(pdf, 7, "References")
    for authors, text in REFERENCES_LIST:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*BLACK)
        pdf.multi_cell(0, 5.8, authors, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5.8, "    " + text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)


def appendices_page(pdf):
    pdf.add_page()
    section_heading(pdf, 8, "Appendices")
    body(pdf, "All figures referenced in this paper are embedded inline within "
         "their respective sections. Source code and raw data files are available "
         "in the project repository.")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*ACCENT)
    pdf.set_text_color(*WHITE)
    for w, h in [(28, 7), (52, 7), (85, 7)]:
        pdf.cell(w, h, ["Figure", "File", "Description"][[28, 52, 85].index(w)],
                 border=1, fill=True, align="C")
    pdf.ln()
    items = [
        ("Figure 1", "fig_eda.png",               "Portfolio returns, rolling vol, VIX with stress shading"),
        ("Figure 2", "fig_hmm_states.png",         "HMM posterior state probabilities"),
        ("Figure 3", "fig_bcp_regimes.png",        "Bayesian changepoint (PELT) segmentation"),
        ("Figure 4", "fig_loss_distributions.png", "Loss distributions by regime + QQ-plot"),
        ("Figure 5", "fig_tail_dependence.png",    "Pairwise tail dependence heatmap (Student-t copula)"),
        ("Figure 6", "fig_cvar_comparison.png",    "CVaR forecast comparison, 2018-2024"),
        ("Figure 7", "fig_mean_excess.png",        "Mean excess plot (GPD validation)"),
        ("Figure 8", "fig_violations.png",         "CVaR violations by model, 2018-2024"),
    ]
    for ri, (fig, fname, desc) in enumerate(items):
        pdf.set_fill_color(*ROW_EVEN) if ri % 2 == 0 else pdf.set_fill_color(*ROW_ODD)
        pdf.set_text_color(*BLACK)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(28, 6, fig,   border=1, fill=True, align="C")
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(52, 6, fname, border=1, fill=True, align="C")
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(85, 6, desc,  border=1, fill=True, align="L")
        pdf.ln()


def disclaimer_page(pdf):
    pdf.add_page()
    pdf.set_draw_color(*LGRAY)
    pdf.set_line_width(0.3)
    pdf.rect(pdf.l_margin, pdf.get_y(), USABLE_W, 8, "")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*DARK_BLUE)
    pdf.cell(0, 8, "Disclaimer", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*DARK_BLUE)
    pdf.set_line_width(0.4)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(*GRAY)
    pdf.multi_cell(0, 5, DISCLAIMER, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


# ── main ─────────────────────────────────────────────────────────────────────

def build(github_url=""):
    pdf = Paper(orientation="P", unit="mm", format="Letter")
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.set_auto_page_break(auto=True, margin=18)

    title_page(pdf, github_url)

    pdf.add_page(); section_heading(pdf, 2, "Introduction");  render_blocks(pdf, INTRO_BLOCKS)
    pdf.add_page(); section_heading(pdf, 3, "Methodology");   render_blocks(pdf, METHOD_BLOCKS)
    pdf.add_page(); section_heading(pdf, 4, "Results");       render_blocks(pdf, RESULTS_BLOCKS)
    pdf.add_page(); section_heading(pdf, 5, "Discussion");    render_blocks(pdf, DISCUSSION_BLOCKS)
    pdf.add_page(); section_heading(pdf, 6, "Conclusion");    render_blocks(pdf, CONCLUSION_BLOCKS)

    references_page(pdf)
    appendices_page(pdf)
    disclaimer_page(pdf)

    pdf.output(OUT)
    print(f"Saved: {OUT}  ({pdf.page} pages)")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else ""
    build(github_url=url)
