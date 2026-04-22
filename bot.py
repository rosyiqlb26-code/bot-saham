"""
Bot Telegram Analisis Saham IDX
Dibuat untuk: Trading Saham Harian style
Perintah: /chart [TICKER]
"""

import logging
import io
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ============================================================
# KONFIGURASI — GANTI TOKEN DI SINI
# ============================================================
TOKEN = "8657690045:AAFK0pmlQqE6VZMZIMT2biYs_egc3bhOygc"

# ============================================================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# AMBIL DATA SAHAM
# ─────────────────────────────────────────────────────────────
def get_stock_data(ticker: str):
    """Ambil data dari Yahoo Finance. Saham IDX pakai suffix .JK"""
    ticker_upper = ticker.upper().replace(".JK", "")
    ticker_yf = ticker_upper + ".JK"

    try:
        stock = yf.Ticker(ticker_yf)
        df = stock.history(period="6mo", auto_adjust=True)
        if df.empty or len(df) < 30:
            return None, None
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df, ticker_upper
    except Exception as e:
        logger.error(f"Gagal ambil data {ticker_yf}: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────
# HITUNG INDIKATOR
# ─────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Moving Average
    df['EMA5']    = df['Close'].ewm(span=5,  adjust=False).mean()
    df['MA20']    = df['Close'].rolling(20).mean()
    df['MA20Vol'] = df['Volume'].rolling(20).mean()

    # MACD (12, 26, 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']   = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist']   = df['MACD'] - df['Signal']

    # MFI (14)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']
    pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg = mf.where(tp <= tp.shift(1), 0).rolling(14).sum()
    df['MFI'] = 100 - (100 / (1 + pos / neg.replace(0, np.nan))).fillna(50)

    return df


# ─────────────────────────────────────────────────────────────
# HITUNG SUPPORT & RESISTANCE (Pivot Point Classic)
# ─────────────────────────────────────────────────────────────
def get_sr_levels(df: pd.DataFrame):
    h = df['High'].iloc[-1]
    l = df['Low'].iloc[-1]
    c = df['Close'].iloc[-1]
    pivot = (h + l + c) / 3
    r1 = 2 * pivot - l
    r2 = pivot + (h - l)
    s1 = 2 * pivot - h
    s2 = pivot - (h - l)
    return r2, r1, s1, s2


# ─────────────────────────────────────────────────────────────
# BUAT CHART
# ─────────────────────────────────────────────────────────────
def generate_chart(df_full: pd.DataFrame, ticker: str) -> io.BytesIO:
    df = calculate_indicators(df_full)
    df = df.tail(65).copy()         # Tampilkan ~3 bulan terakhir
    df.reset_index(inplace=True)    # index jadi kolom 'Date'

    r2, r1, s1, s2 = get_sr_levels(df)
    n = len(df)
    xs = np.arange(n)

    # ── Setup figure ──────────────────────────────────────────
    BG   = '#FAFAFA'
    GRID = '#E8E8E8'
    fig  = plt.figure(figsize=(13, 9), facecolor=BG)

    # 4 panel: Price | Volume | MACD | MFI
    gs  = gridspec.GridSpec(4, 1, figure=fig,
                            height_ratios=[5, 1.8, 1.8, 1.2],
                            hspace=0.04)
    ax1 = fig.add_subplot(gs[0])   # Price
    ax2 = fig.add_subplot(gs[1], sharex=ax1)   # Volume
    ax3 = fig.add_subplot(gs[2], sharex=ax1)   # MACD
    ax4 = fig.add_subplot(gs[3], sharex=ax1)   # MFI

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(BG)
        ax.tick_params(colors='#333333', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.yaxis.tick_right()
        ax.grid(color=GRID, linewidth=0.4, alpha=0.8)

    # ── Panel 1: Candlestick ──────────────────────────────────
    GREEN = '#26a69a'   # bullish
    RED   = '#ef5350'   # bearish

    for i, row in df.iterrows():
        col = GREEN if row['Close'] >= row['Open'] else RED
        ax1.plot([i, i], [row['Low'], row['High']], color=col, lw=0.9, zorder=2)
        h = max(abs(row['Close'] - row['Open']), row['High'] * 0.001)
        b = min(row['Close'], row['Open'])
        ax1.bar(i, h, bottom=b, color=col, width=0.6, zorder=3)

    ax1.plot(xs, df['EMA5'],  color='#00b0f0', lw=1.3, label='EMA5', zorder=4)
    ax1.plot(xs, df['MA20'],  color='#FFA500', lw=1.3, label='MA20', zorder=4)

    # S/R Lines & shading
    ax1.axhline(r2, color='#9e9e9e', lw=0.8, ls='--')
    ax1.axhline(r1, color='#9e9e9e', lw=0.8, ls='--')
    ax1.axhline(s1, color='#9e9e9e', lw=0.8, ls='--')
    ax1.axhline(s2, color='#9e9e9e', lw=0.8, ls='--')
    ax1.axhspan(r1, r2, alpha=0.08, color='#ef5350')  # resistance zone

    # Labels kanan
    price_now = df['Close'].iloc[-1]
    ema5_now  = df['EMA5'].iloc[-1]
    ma20_now  = df['MA20'].iloc[-1]
    xl        = n + 0.8

    def label(ax, y, text, fc):
        ax.annotate(text, xy=(n-1, y), xytext=(xl, y),
                    xycoords=('data','data'), textcoords=('data','data'),
                    va='center', ha='left', fontsize=6, color='white',
                    bbox=dict(boxstyle='round,pad=0.25', fc=fc, alpha=0.85))

    label(ax1, r2,       f'R2 {int(r2)}',        '#757575')
    label(ax1, r1,       f'R1 {int(r1)}',        '#9e9e9e')
    label(ax1, ema5_now, f'EMA5 {int(ema5_now)}','#00b0f0')
    label(ax1, ma20_now, f'MA20 {int(ma20_now)}','#FFA500')
    label(ax1, price_now,f'{int(price_now)}',    '#424242')
    label(ax1, s1,       f'S1 {int(s1)}',        '#9e9e9e')
    label(ax1, s2,       f'S2 {int(s2)}',        '#757575')

    ax1.set_xlim(-1, n + 5)
    plo, phi = df['Low'].min()*0.97, df['High'].max()*1.03
    ax1.set_ylim(plo, phi)

    # Judul & watermark
    from datetime import datetime
    ax1.set_title(f'{ticker}', color='#212121', fontsize=13,
                  fontweight='bold', loc='left', pad=6)
    fig.text(0.5, 0.02, datetime.now().strftime('%d %b %Y'),
             ha='center', fontsize=7, color='#9e9e9e')
    ax1.text(0.5, 0.5, 'Trading Saham Harian',
             transform=ax1.transAxes, fontsize=18,
             color='gray', alpha=0.07, ha='center', va='center',
             fontweight='bold', rotation=0)

    # ── Panel 2: Volume ───────────────────────────────────────
    for i, row in df.iterrows():
        col = GREEN if row['Close'] >= row['Open'] else RED
        ax2.bar(i, row['Volume'], color=col, width=0.6, alpha=0.75)
    ax2.plot(xs, df['MA20Vol'], color='#5c6bc0', lw=1.1)

    ma20v = df['MA20Vol'].iloc[-1]
    label_txt = f"MA20 {ma20v/1e6:.1f}M" if ma20v >= 1e6 else f"MA20 {int(ma20v/1e3)}K"
    ax2.annotate(label_txt, xy=(n-1, ma20v), xytext=(xl, ma20v),
                 xycoords='data', textcoords='data',
                 va='center', ha='left', fontsize=6, color='white',
                 bbox=dict(boxstyle='round,pad=0.25', fc='#5c6bc0', alpha=0.85))

    # Label
    ax2.text(0.01, 0.92, 'VOLUME', transform=ax2.transAxes,
             fontsize=6, color='#555555', va='top')
    ax2.set_xlim(-1, n + 5)

    # ── Panel 3: MACD ─────────────────────────────────────────
    for i, row in df.iterrows():
        col = GREEN if row['Hist'] >= 0 else RED
        ax3.bar(i, row['Hist'], color=col, width=0.6, alpha=0.8)
    ax3.plot(xs, df['MACD'],   color='#1e88e5', lw=1.1, label='MACD')
    ax3.plot(xs, df['Signal'], color='#e53935', lw=1.1, label='Signal')
    ax3.axhline(0, color='#9e9e9e', lw=0.5)

    ml, sl, hl = df['MACD'].iloc[-1], df['Signal'].iloc[-1], df['Hist'].iloc[-1]
    label(ax3, sl, f'SIG {sl:.2f}',  '#e53935')
    label(ax3, ml, f'MACD {ml:.2f}', '#1e88e5')
    label(ax3, hl, f'HIST {hl:.2f}', GREEN if hl>=0 else RED)

    ax3.text(0.01, 0.92, 'MACD (12,26,9)', transform=ax3.transAxes,
             fontsize=6, color='#555555', va='top')
    ax3.set_xlim(-1, n + 5)

    # ── Panel 4: MFI ──────────────────────────────────────────
    ax4.plot(xs, df['MFI'], color='#8e24aa', lw=1.2)
    ax4.axhline(80, color=RED,   lw=0.6, ls='--', alpha=0.7)
    ax4.axhline(20, color=GREEN, lw=0.6, ls='--', alpha=0.7)
    ax4.set_ylim(0, 100)

    mfi_now = df['MFI'].iloc[-1]
    label(ax4, mfi_now, f'MFI {mfi_now:.1f}', '#8e24aa')

    ax4.text(0.01, 0.92, 'MFI (14)', transform=ax4.transAxes,
             fontsize=6, color='#555555', va='top')
    ax4.set_xlim(-1, n + 5)

    # ── X-axis labels (hanya di panel bawah) ─────────────────
    ticks, labels = [], []
    prev_m = None
    for i, row in df.iterrows():
        d = row['Date']
        m = d.strftime('%b')
        if m != prev_m or d.day <= 5:
            if m != prev_m:
                ticks.append(i)
                labels.append(f"{d.day:02d} {m}")
                prev_m = m
    ax4.set_xticks(ticks)
    ax4.set_xticklabels(labels, color='#333333', fontsize=7)
    for ax in [ax1, ax2, ax3]:
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.tight_layout(pad=0.8)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150,
                bbox_inches='tight', facecolor=BG)
    buf.seek(0)
    plt.close(fig)
    return buf


# ─────────────────────────────────────────────────────────────
# BUAT TEKS ANALISIS
# ─────────────────────────────────────────────────────────────
def generate_analysis(df_full: pd.DataFrame, ticker: str) -> str:
    df = calculate_indicators(df_full)
    r2, r1, s1, s2 = get_sr_levels(df)

    price  = int(df['Close'].iloc[-1])
    ema5   = int(df['EMA5'].iloc[-1])
    ma20   = int(df['MA20'].iloc[-1])
    mfi    = df['MFI'].iloc[-1]
    macd   = df['MACD'].iloc[-1]
    hist   = df['Hist'].iloc[-1]

    # Hitung perubahan harga 5 hari & 20 hari
    ret5  = (df['Close'].iloc[-1] / df['Close'].iloc[-6]  - 1) * 100 if len(df) >= 6  else 0
    ret20 = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100 if len(df) >= 21 else 0

    # Volume
    vol5  = df['Volume'].tail(5).mean()
    vol20 = df['Volume'].tail(20).mean()
    vol_ratio = vol5 / vol20 if vol20 > 0 else 1

    vol_txt = ("Volume tinggi 🔥" if vol_ratio > 1.5
               else "Volume normal" if vol_ratio > 0.9
               else "Volume masih tipis")

    # ── SCORING SISTEM ─────────────────────────────────────────
    # Setiap kondisi beri skor: positif = bullish, negatif = bearish
    score = 0
    warnings = []
    positives = []

    # 1. Posisi harga vs EMA5 & MA20
    if price > ema5 > ma20:
        score += 3
        positives.append("Harga di atas EMA5 & MA20 ✅")
    elif price > ma20:
        score += 1
        positives.append("Harga di atas MA20 ✅")
    elif price < ema5 < ma20:
        score -= 3
        warnings.append("Harga di bawah EMA5 & MA20 ❌")
    elif price < ma20:
        score -= 1
        warnings.append("Harga di bawah MA20 ❌")

    # 2. MACD
    if macd > 0 and hist > 0:
        score += 2
        positives.append("MACD positif & histogram naik ✅")
    elif macd > 0 and hist < 0:
        score += 0
        warnings.append("MACD positif tapi momentum melemah ⚠️")
    elif macd < 0 and hist < 0:
        score -= 2
        warnings.append("MACD negatif & histogram turun ❌")
    elif macd < 0 and hist > 0:
        score += 0  # ada perbaikan tapi belum konfirmasi
        positives.append("MACD mulai membaik (belum konfirmasi)")

    # 3. MFI
    if mfi > 80:
        score -= 1
        warnings.append(f"MFI {mfi:.0f} — Overbought, rawan koreksi ⚠️")
    elif mfi < 20:
        score += 1
        positives.append(f"MFI {mfi:.0f} — Oversold, potensi rebound 🟢")
    elif mfi > 50:
        score += 1
        positives.append(f"Arus kas ngalir 👍 (MFI {mfi:.0f})")
    else:
        score -= 1
        warnings.append(f"Arus kas lemah (MFI {mfi:.0f})")

    # 4. Performa harga
    if ret5 < -10:
        score -= 2
        warnings.append(f"Turun {ret5:.1f}% dalam 5 hari ❌")
    elif ret5 < -5:
        score -= 1
        warnings.append(f"Turun {ret5:.1f}% dalam 5 hari ⚠️")
    elif ret5 > 5:
        score += 1
        positives.append(f"Naik {ret5:.1f}% dalam 5 hari 📈")

    # 5. Volume saat turun = sinyal distribusi
    if vol_ratio > 1.5 and ret5 < -3:
        score -= 2
        warnings.append("Volume tinggi saat harga turun — distribusi ❌")
    elif vol_ratio > 1.5 and ret5 > 3:
        score += 1
        positives.append("Volume tinggi saat harga naik — akumulasi 🔥")

    # ── KEPUTUSAN BERDASARKAN SCORE ───────────────────────────
    if score >= 4:
        verdict     = "LAYAK DIPERTIMBANGKAN ✅"
        verdict_emoji = "🟢"
        show_entry  = True
        catatan     = "Kondisi teknikal bagus. Tetap disiplin cut loss ya!"
    elif score >= 1:
        verdict     = "NETRAL — HATI-HATI ⚠️"
        verdict_emoji = "🟡"
        show_entry  = True
        catatan     = "Kondisi campur aduk. Cicil kecil dulu, jangan FOMO."
    elif score >= -1:
        verdict     = "WAIT & SEE — BELUM AMAN ⚠️"
        verdict_emoji = "🟠"
        show_entry  = False
        catatan     = "Tunggu konfirmasi lebih lanjut sebelum masuk."
    else:
        verdict     = "HINDARI DULU — KONDISI BURUK 🚫"
        verdict_emoji = "🔴"
        show_entry  = False
        catatan     = "Jangan masuk dulu! Tunggu tanda-tanda pembalikan arah."

    # ── SUSUN TEKS ────────────────────────────────────────────
    # Trend label
    if price > ema5 > ma20:
        trend = "Bullish Kuat"
    elif price > ema5:
        trend = "Bullish"
    elif price < ema5 < ma20:
        trend = "Bearish"
    elif price < ma20:
        trend = "Bearish / Sideways"
    else:
        trend = "Sideways"

    text = (
        f"{verdict_emoji} *{ticker} {price}*\n"
        f"Trend : _{trend}_\n"
        f"Sinyal : *{verdict}*\n\n"
    )

    if warnings:
        text += "*Peringatan :*\n"
        for w in warnings:
            text += f"• {w}\n"
        text += "\n"

    if positives:
        text += "*Positif :*\n"
        for p in positives:
            text += f"• {p}\n"
        text += "\n"

    text += f"• {vol_txt}\n\n"

    if show_entry:
        entry_lo = int(min(s1, ema5) * 0.99)
        entry_hi = int(ema5)
        tp1      = int(r1)
        tp2      = int(r2)
        text += (
            f"*Strategi :*\n"
            f"- Entry dekat {entry_lo} – {entry_hi} atau EMA5/MA20\n"
            f"- Cicil dulu → average di bawah\n"
            f"- Target R1 {tp1} → R2 {tp2}\n"
            f"- Cut loss jika tutup di bawah S1 {int(s1)}\n\n"
        )
    else:
        text += (
            f"*Strategi :*\n"
            f"- SKIP dulu, jangan masuk!\n"
            f"- Masuk hanya jika harga kembali di atas EMA5 {ema5}\n"
            f"- Pantau apakah bisa tutup di atas MA20 {ma20}\n\n"
        )

    text += (
        f"*Support :* S1 {int(s1)} | S2 {int(s2)}\n"
        f"*Resistance :* R1 {int(r1)} | R2 {int(r2)}\n\n"
        f"_{catatan}_"
    )

    return text


# ─────────────────────────────────────────────────────────────
# HANDLER TELEGRAM
# ─────────────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 *Selamat datang di Bot Analisis Saham IDX!*\n\n"
        "📊 Perintah tersedia:\n"
        "• /chart \\[kode\\] — Chart & analisis teknikal\n"
        "• /help — Panduan lengkap\n\n"
        "Contoh: `/chart BBCA` atau `/chart TLKM`\n\n"
        "Selamat trading\\! 🚀"
    )
    await update.message.reply_text(msg, parse_mode='MarkdownV2')


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "📚 *Panduan Bot Saham*\n\n"
        "*Cara pakai:*\n"
        "`/chart BBCA` — Analisis saham BBCA\n"
        "`/chart GOTO` — Analisis saham GOTO\n\n"
        "*Indikator di chart:*\n"
        "• Candlestick harian\n"
        "• EMA5 \\(biru\\) & MA20 \\(oranye\\)\n"
        "• Support & Resistance \\(S1, S2, R1, R2\\)\n"
        "• Volume \\+ MA20 Volume\n"
        "• MACD \\(12,26,9\\)\n"
        "• Money Flow Index / MFI \\(14\\)\n\n"
        "_Data dari Yahoo Finance\\. Untuk live intraday, gunakan data broker\\._"
    )
    await update.message.reply_text(msg, parse_mode='MarkdownV2')


async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "❌ Format: /chart \\[kode\\_saham\\]\n"
            "Contoh: `/chart BBCA`",
            parse_mode='MarkdownV2'
        )
        return

    ticker = context.args[0].strip()
    loading = await update.message.reply_text(f"⏳ Menganalisis *{ticker.upper()}*...",
                                               parse_mode='Markdown')
    try:
        df, clean_ticker = get_stock_data(ticker)

        if df is None:
            await loading.edit_text(
                f"❌ Saham *{ticker.upper()}* tidak ditemukan\\.\n"
                f"Pastikan kode benar, contoh: BBCA, TLKM, GOTO",
                parse_mode='MarkdownV2'
            )
            return

        chart_buf = generate_chart(df, clean_ticker)
        analysis  = generate_analysis(df, clean_ticker)

        await loading.delete()
        await update.message.reply_photo(
            photo=chart_buf,
            caption=analysis,
            parse_mode='Markdown'
        )

    except Exception as e:
        logger.exception(f"Error saat analisis {ticker}")
        await loading.edit_text(f"❌ Terjadi error saat analisis {ticker.upper()}. Coba lagi.")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    if TOKEN == "MASUKKAN_TOKEN_BOT_ANDA_DI_SINI":
        print("⚠️  STOP! Ganti TOKEN di baris 18 dengan token dari BotFather!")
        return

    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("chart", cmd_chart))

    print("🤖 Bot berjalan! Buka Telegram dan kirim /start")
    print("   Tekan Ctrl+C untuk berhenti.\n")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()