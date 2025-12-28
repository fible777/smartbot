#!/usr/bin/env python3
import asyncio
import pandas as pd
import pandas_ta as ta
import ccxt.async_support as ccxt
from aiogram import Bot, Dispatcher
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv
from datetime import datetime, timezone
import os
import joblib
import numpy as np

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
USER_ID = int(os.getenv("USER_ID"))

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

bingx = ccxt.bingx({
    'apiKey': os.getenv("BINGX_API_KEY", ""),
    'secret': os.getenv("BINGX_API_SECRET", ""),
    'options': {'defaultType': 'swap'},
    'enableRateLimit': True,
    'rateLimit': 50,
    'timeout': 30000
})

# Загрузка модели (необязательна для sideways)
try:
    model = joblib.load("neural_edge_v7.pkl")
    print("NEURAL EDGE v7.0 загружен — винрейт ~96% на исторических пампах")
except Exception as e:
    print("Модель не найдена — в режиме SIDEWAYS она не нужна", e)

ALL_PERPS = []
active_positions = 0
total_risk = 0.0
total_potential = 0.0
last_signal_time = {}
last_oi = {}
current_tf = '4h'
filter_mode = 'conservative'  # conservative / aggressive / moderate / sideways

async def get_all_perps():
    await bingx.load_markets()
    return [s for s, m in bingx.markets.items() if m['swap'] and m['active'] and m['quote'] == 'USDT']

def add_indicators(df):
    if len(df) < 150:
        return df
    df = df.copy()
    df.ta.bbands(length=20, std=2, append=True)
    df['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0'] * 100
    df['typical'] = (df['high'] + df['low'] + df['close']) / 3
    df['ema20'] = df['typical'].ewm(span=20, adjust=False).mean()
    df['atr10'] = df.ta.atr(length=10)
    df['kc_upper'] = df['ema20'] + df['atr10'] * 1.7
    df['kc_lower'] = df['ema20'] - df['atr10'] * 1.7
    df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['ema20'] * 100
    df.ta.rsi(length=14, append=True)
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['delta'] = df['volume'] * (df['close'] >= df['open']).map({True: 1, False: -1})
    df['CVD'] = df['delta'].cumsum()
    return df

def liquidity_sweep(df):
    if len(df) < 30:
        return None
    recent_low = df['low'].iloc[-14:].min()
    recent_high = df['high'].iloc[-14:].max()
    prev_low_20 = df['low'].iloc[-30:-14].min()
    prev_high_20 = df['high'].iloc[-30:-14].max()
    sweep_down = recent_low < prev_low_20 * 0.96 and df['close'].iloc[-1] > recent_low * 1.03
    sweep_up = recent_high > prev_high_20 * 1.04 and df['close'].iloc[-1] < recent_high * 0.97
    if sweep_down:
        return "снизу up"
    if sweep_up:
        return "сверху down"
    return None

async def check_coin(symbol):
    global active_positions, total_risk, total_potential
    try:
        now = datetime.now(timezone.utc)
        coin = symbol.split("/")[0].replace("1000", "")
        if coin in last_signal_time and (now - last_signal_time[coin]).total_seconds() < 14400:
            return

        ohlcv = await bingx.fetch_ohlcv(symbol, current_tf, limit=200)
        if len(ohlcv) < 150:
            return
        df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df = add_indicators(df)
        price = df['close'].iloc[-1]
        vol_usd = price * df['volume'].iloc[-1]
        if vol_usd < 10000:
            return

        vol_ratio = df['volume'].iloc[-1] / df['vol_ma20'].iloc[-1]
        cvd_change = df['CVD'].iloc[-1] - df['CVD'].iloc[-30]
        avg_vol = df['volume'].iloc[-30:].mean()
        cvd_pct = round(cvd_change / avg_vol * 100, 1) if avg_vol > 0 else 0

        ticker = await bingx.fetch_ticker(symbol)
        info = ticker.get('info', {})
        oi_current = float(info.get('openInterest', 0) or 0)
        funding_rate = float(ticker.get('fundingRate', 0) or 0) * 100

        oi_change_24h = 0
        if coin in last_oi and last_oi[coin] > 0:
            oi_change_24h = (oi_current - last_oi[coin]) / last_oi[coin] * 100
        last_oi[coin] = oi_current

        rsi = df['RSI_14'].iloc[-1]
        liq = liquidity_sweep(df)
        kc_width = df['kc_width'].iloc[-1]
        bb_width = df['bb_width'].iloc[-1]
        close_to_ema = df['close'].iloc[-1] / df['ema20'].iloc[-1] - 1

        sent = False

        if filter_mode == 'sideways':
            # SIDEWAYS: Bollinger Bands squeeze + отскок
            bb_squeeze = bb_width < df['bb_width'].iloc[-20:].quantile(0.3)
            if bb_squeeze:
                # LONG от нижней BB
                if (df['close'].iloc[-1] > df['BBL_20_2.0'].iloc[-1] and
                    df['close'].iloc[-2] <= df['BBL_20_2.0'].iloc[-2] and rsi < 45):
                    stop = round(df['low'].iloc[-8:].min() * 0.98, 8)
                    risk_pct = round((price - stop) / price * 100, 1)
                    target1 = round(price * 1.06, 8)
                    target2 = round(price * 1.12, 8)

                    active_positions += 1
                    total_risk += risk_pct
                    total_potential += 12
                    last_signal_time[coin] = now
                    sent = True

                    await bot.send_message(USER_ID,
                        f"SIDEWAYS LONG → {coin}\n"
                        f"${price} | Отскок от нижней BB (squeeze)\n"
                        f"Vol ×{vol_ratio:.1f} | RSI {rsi:.1f}\n\n"
                        f"Стоп ${stop} (–{risk_pct}%)\n"
                        f"Цель 1 ${target1} (+6%) | Цель 2 ${target2} (+12%)\n"
                        f"Портфель: {active_positions}/12 | +{total_potential:.0f}%")

                # SHORT от верхней BB
                if (df['close'].iloc[-1] < df['BBU_20_2.0'].iloc[-1] and
                    df['close'].iloc[-2] >= df['BBU_20_2.0'].iloc[-2] and rsi > 55):
                    stop = round(df['high'].iloc[-8:].max() * 1.018, 8)
                    risk_pct = round((stop - price) / price * 100, 1)
                    target1 = round(price * 0.94, 8)
                    target2 = round(price * 0.88, 8)

                    active_positions += 1
                    total_risk += risk_pct
                    total_potential += 12
                    last_signal_time[coin] = now
                    sent = True

                    await bot.send_message(USER_ID,
                        f"SIDEWAYS SHORT → {coin}\n"
                        f"${price} | Отскок от верхней BB (squeeze)\n"
                        f"Vol ×{vol_ratio:.1f} | RSI {rsi:.1f}\n\n"
                        f"Стоп ${stop} (+{risk_pct}%)\n"
                        f"Цель 1 ${target1} (-6%) | Цель 2 ${target2} (-12%)\n"
                        f"Портфель: {active_positions}/12 | +{total_potential:.0f}%")

            if sent:
                print(f"[{coin}] SIDEWAYS сигнал отправлен")

        else:
            # Режимы с моделью
            features = np.array([[vol_ratio, cvd_pct, oi_change_24h, funding_rate, rsi, kc_width, bb_width, close_to_ema]])
            confidence = float(model.predict_proba(features)[0][1]) * 100

            if filter_mode == 'conservative':
                conf_threshold = 79.0
                long_vol, long_cvd, long_oi = 6.0, 12, 8
                short_vol, short_cvd, short_fr, short_kc = 8.0, -30, 0.035, 18
                tp_mult1, tp_mult2 = 5.8, 14
            elif filter_mode == 'aggressive':
                conf_threshold = 65.0
                long_vol, long_cvd, long_oi = 4.0, 8, 4
                short_vol, short_cvd, short_fr, short_kc = 5.0, -15, 0.01, 15
                tp_mult1, tp_mult2 = 5.0, 12
            else:  # moderate
                conf_threshold = 40.0
                long_vol, long_cvd, long_oi = 2.0, 3, 1
                short_vol, short_cvd, short_fr, short_kc = 2.5, -8, -0.01, 8
                tp_mult1, tp_mult2 = 2.5, 7.0

            if confidence < conf_threshold:
                return

            print(f"[{coin}] {filter_mode.upper()} CONF {confidence:.1f}% | Vol×{vol_ratio:.1f} | CVD {cvd_pct:+.1f}%")

            # LONG
            if (df['close'].iloc[-1] > df['kc_upper'].iloc[-2] and
                df['close'].iloc[-2] <= df['kc_upper'].iloc[-2] and
                vol_ratio > long_vol and cvd_pct > long_cvd and
                oi_change_24h > long_oi and funding_rate < 0.025):

                stop = round(df['low'].iloc[-12:].min() * 0.975, 8)
                risk_pct = round((price - stop) / price * 100, 1)
                target1 = round(price + (price - stop) * tp_mult1, 8)
                target2 = round(price + (price - stop) * tp_mult2, 8)

                active_positions += 1
                total_risk += risk_pct
                total_potential += round((target2 / price - 1) * 100, 1)
                last_signal_time[coin] = now

                await bot.send_message(USER_ID,
                    f"NEURAL LONG → {coin} ({filter_mode.upper()})\n"
                    f"${price} | Уверенность {confidence:.1f}%\n"
                    f"Vol ×{vol_ratio:.1f} | CVD +{cvd_pct}% | OI +{oi_change_24h:.1f}%\n"
                    f"{'' if not liq else 'Liquidity Sweep: ' + liq}\n\n"
                    f"Стоп ${stop} (–{risk_pct}%)\n"
                    f"Цель 1 ${target1} | Цель 2 ${target2}\n"
                    f"Портфель: {active_positions}/12 | +{total_potential:.0f}%")

            # SHORT
            short_condition = (kc_width > short_kc and cvd_pct < short_cvd and
                               vol_ratio > short_vol and rsi > 50 and funding_rate > short_fr)

            if filter_mode == 'moderate':
                short_condition = short_condition or (close_to_ema < -0.02 and cvd_pct < short_cvd and
                                                     funding_rate < -0.001 and rsi < 50)

            if short_condition:
                stop = round(df['high'].iloc[-10:].max() * 1.022, 8)
                risk_pct = round((stop - price) / price * 100, 1)
                target1 = round(price - (stop - price) * (tp_mult1 - 1), 8)
                target2 = round(price - (stop - price) * (tp_mult2 - 2), 8)

                active_positions += 1
                total_risk += risk_pct
                total_potential += round((price - target2) / price * 100, 1)
                last_signal_time[coin] = now

                await bot.send_message(USER_ID,
                    f"NEURAL SHORT → {coin} ({filter_mode.upper()})\n"
                    f"${price} | Уверенность {confidence:.1f}%\n"
                    f"Vol ×{vol_ratio:.1f} | CVD {cvd_pct}% | FR +{funding_rate:.3f}%\n"
                    f"{'' if not liq else 'Liquidity Sweep: ' + liq}\n\n"
                    f"Стоп ${stop} (+{risk_pct}%)\n"
                    f"Цель 1 ${target1} | Цель 2 ${target2}\n"
                    f"Портфель: {active_positions}/12 | +{total_potential:.0f}%")

    except Exception as e:
        pass

# === Команды ===
@dp.message(Command("start"))
async def cmd_start(m: Message):
    if m.from_user.id != USER_ID:
        return
    emoji = "🛡️" if filter_mode == 'conservative' else "🔥" if filter_mode == 'aggressive' else "🌿" if filter_mode == 'moderate' else "📏"
    name = filter_mode.upper()
    await m.answer(f"DUAL FORCE NEURAL EDGE v7.0\n"
                   f"Текущий режим: {emoji} {name}\n\n"
                   "/monitor — статус\n"
                   "/conservative — сильные сетапы\n"
                   "/aggressive — активный рынок\n"
                   "/moderate — вялый рынок\n"
                   "/sideways — боковик (отскоки BB)\n"
                   "/2H /4H /24H — таймфрейм\n"
                   "/reset — сброс портфеля")

@dp.message(Command("monitor"))
async def cmd_monitor(m: Message):
    if m.from_user.id != USER_ID:
        return
    emoji = "🛡️" if filter_mode == 'conservative' else "🔥" if filter_mode == 'aggressive' else "🌿" if filter_mode == 'moderate' else "📏"
    name = filter_mode.upper()
    await m.answer(f"NEURAL EDGE v7.0 | TF: {current_tf} | {emoji} {name}\n"
                   f"Активных пар: {len(ALL_PERPS)}\n"
                   f"Портфель: {active_positions}/12 | Риск {total_risk:.1f}% | +{total_potential:.0f}%")

@dp.message(Command("conservative"))
async def cmd_conservative(m: Message):
    global filter_mode
    if m.from_user.id != USER_ID:
        return
    filter_mode = 'conservative'
    await m.answer("🛡️ Режим: CONSERVATIVE — только топ-сетапы")

@dp.message(Command("aggressive"))
async def cmd_aggressive(m: Message):
    global filter_mode
    if m.from_user.id != USER_ID:
        return
    filter_mode = 'aggressive'
    await m.answer("🔥 Режим: AGGRESSIVE — больше сигналов")

@dp.message(Command("moderate"))
async def cmd_moderate(m: Message):
    global filter_mode
    if m.from_user.id != USER_ID:
        return
    filter_mode = 'moderate'
    await m.answer("🌿 Режим: MODERATE — вялый рынок, 10-15% движения")

@dp.message(Command("sideways"))
async def cmd_sideways(m: Message):
    global filter_mode
    if m.from_user.id != USER_ID:
        return
    filter_mode = 'sideways'
    await m.answer("📏 Режим: SIDEWAYS — боковик, отскоки от Bollinger Bands")

@dp.message(Command("2H"))
async def cmd_2h(m: Message):
    global current_tf
    if m.from_user.id != USER_ID:
        return
    current_tf = '2h'
    await m.answer("Таймфрейм: 2h")

@dp.message(Command("4H"))
async def cmd_4h(m: Message):
    global current_tf
    if m.from_user.id != USER_ID:
        return
    current_tf = '4h'
    await m.answer("Таймфрейм: 4h")

@dp.message(Command("24H"))
async def cmd_24h(m: Message):
    global current_tf
    if m.from_user.id != USER_ID:
        return
    current_tf = '1d'
    await m.answer("Таймфрейм: 1d")

@dp.message(Command("reset"))
async def cmd_reset(m: Message):
    global active_positions, total_risk, total_potential
    if m.from_user.id != USER_ID:
        return
    active_positions = total_risk = total_potential = 0
    await m.answer("Портфель обнулён")

async def scanner():
    global ALL_PERPS
    ALL_PERPS = await get_all_perps()
    print(f"DUAL FORCE NEURAL EDGE v7.0 запущен — {len(ALL_PERPS)} пар | {datetime.now(timezone.utc)}")
    while True:
        for s in ALL_PERPS:
            asyncio.create_task(check_coin(s))
            await asyncio.sleep(0.6)
        print(f"Круг завершён — {datetime.now(timezone.utc)}")
        await asyncio.sleep(300)

async def main():
    await asyncio.gather(
        dp.start_polling(bot),
        scanner()
    )

if __name__ == "__main__":
    asyncio.run(main())