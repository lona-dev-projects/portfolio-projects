
#property version   "1.0"

#include <Trade\Trade.mqh>

CTrade trade;
input group              "General Settings"
input int                InpWarmupBars = 50;
input string             InpSymbolsToTrade = "EURUSD,GBPUSD,XAUUSD";
input ulong              InpMagicNumber = 12345;
input int                InpSlippage = 5;
input double             InpRiskPercentPerTrade = 1.5;
input ENUM_TIMEFRAMES    InpSignalTimeframe = PERIOD_M5;
input ENUM_TIMEFRAMES    InpContextTimeframe = PERIOD_M15;

input group              "Trading Hours (Optional)"
input bool               InpUseTradingHours = false;
input int                InpStartHour = 0;
input int                InpStartMinute = 0;
input int                InpEndHour = 23;
input int                InpEndMinute = 59;

input group              "Core Strategy Filters"
input int                InpSMA_Long_Period = 30;
input int                InpADX_Period = 14;
input double             InpADX_Max_Level = 28.0;
input double             InpMinRiskReward = 3.0;

input group              "Volatility Filter"
input int                InpVolatility_ATR_Period = 14;
input double             InpVolatility_Min_ATR_Pips_ContextTF = 5.0;

input group              "Momentum & Confirmation (Signal TF)"
input int                InpRSI_Period = 14;
input double             InpRSI_Long_Min_Level = 55.0;
input double             InpRSI_Short_Max_Level = 45.0;
input int                InpSTOCH_K = 5;
input int                InpSTOCH_D = 3;
input int                InpSTOCH_Slowing = 3;
input double             InpSTOCH_Long_Min_Level_Main = 50.0;
input double             InpSTOCH_Long_Cross_From_Oversold = 25.0;
input double             InpSTOCH_Short_Max_Level_Main = 50.0;
input double             InpSTOCH_Short_Cross_From_Overbought = 75.0;
input double             InpClosePosRatio_Long_Min = 0.60;
input double             InpClosePosRatio_Short_Max = 0.40;

input group              "Candlestick Filters"
input bool               InpUsePinBarFilter = true;
input int                InpDoji_Lookback_Bars = 3;
input int                InpDoji_Max_Allowed = 1;

input group              "Stop Loss & Take Profit"
input int                InpSL_ATR_Period_SignalTF = 14;
input double             InpSL_ATR_Multiplier = 1.5;
input bool               InpUseSwingPointSL = true;
input int                InpSwingPointLookback_SignalTF = 5;

input group              "Trade Management"
input bool               InpUseTrailingStop = true;
input double             InpTrail_ATR_Multiplier = 1.0;
input bool               InpUseBreakeven = true;
input double             InpBreakeven_ATR_Multiplier_Profit = 1.0;
input int                InpMaxHoldTime_Bars_SignalTF = 72;

string G_SymbolsArray[];
int    G_SignalCandleShift = 1;
bool   G_WarmupComplete[]; 

struct SymbolIndicators
{
    int hSMA_Long_ContextTF;
    int hADX_ContextTF;
    int hATR_Volatility_ContextTF;
    int hRSI_SignalTF;
    int hSTOCH_SignalTF;
    int hATR_Management_SignalTF;
};
SymbolIndicators G_SymbolIndicatorHandles[];

struct TradeState
{
    bool     hasOpenTrade;
    ulong    ticket;
    ENUM_ORDER_TYPE tradeType;
    double   openPrice;
    double   initialStopLossPrice;
    double   initialTakeProfitPrice;
    datetime openTime;
    double   lotSize;
};
TradeState G_SymbolTradeStates[];

struct SymbolCache
{
    double point;
    int digits;
    double minLot;
    double maxLot;
    double lotStep;
    double contractSize;
    double tickValue;
    double tickSize;
    double stopLevel;
    string profitCurrency;
    string baseCurrency;
    datetime lastUpdate;
};
SymbolCache G_SymbolCache[];

void CacheSymbolProperties(string symbol, int index)
{
    G_SymbolCache[index].point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    G_SymbolCache[index].digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    G_SymbolCache[index].minLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    G_SymbolCache[index].maxLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    G_SymbolCache[index].lotStep = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    G_SymbolCache[index].contractSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE);
    G_SymbolCache[index].tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE_PROFIT);
    G_SymbolCache[index].tickSize = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    G_SymbolCache[index].stopLevel = (double)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
    G_SymbolCache[index].profitCurrency = SymbolInfoString(symbol, SYMBOL_CURRENCY_PROFIT);
    G_SymbolCache[index].baseCurrency = SymbolInfoString(symbol, SYMBOL_CURRENCY_BASE);
    G_SymbolCache[index].lastUpdate = TimeCurrent();
}

void RefreshSymbolCache(string symbol, int index)
{
    if(TimeCurrent() - G_SymbolCache[index].lastUpdate > 300)
    {
        CacheSymbolProperties(symbol, index);
    }
}

bool CheckWarmupPeriod(string symbol, int symbol_idx)
{
    if(symbol_idx >= ArraySize(G_WarmupComplete)) return false; // Safety check
    if(G_WarmupComplete[symbol_idx]) return true;

    int min_bars_needed = MathMax(InpSMA_Long_Period, InpADX_Period);
    min_bars_needed = MathMax(min_bars_needed, InpRSI_Period);
    min_bars_needed = MathMax(min_bars_needed, InpVolatility_ATR_Period);
    min_bars_needed = MathMax(min_bars_needed, InpSL_ATR_Period_SignalTF);
    min_bars_needed = MathMax(min_bars_needed, InpWarmupBars);

    int signal_bars = iBars(symbol, InpSignalTimeframe);
    int context_bars = iBars(symbol, InpContextTimeframe);

    if(signal_bars < min_bars_needed || context_bars < min_bars_needed)
    {
        Print("Warmup: ", symbol, " needs more bars. Signal:", signal_bars, " Context:", context_bars, " Required:", min_bars_needed);
        return false;
    }
    
    G_WarmupComplete[symbol_idx] = true;
    Print("Warmup period complete for ", symbol);
    return true;
}

bool SelectSymbolWithRetry(string symbol, int max_retries = 3)
{
    for(int retry = 0; retry < max_retries; retry++)
    {
        if(SymbolSelect(symbol, true))
        {
            if(retry > 0) Print("Symbol selection for ", symbol, " succeeded on retry #", retry + 1);
            return true;
        }
        
        Print("Symbol selection failed for ", symbol, " (attempt ", retry + 1, "/", max_retries, "). Error: ", GetLastError());
        Sleep(100 * (int)MathPow(2, retry)); 
        ResetLastError();
    }
    
    Print("CRITICAL: Could not select symbol ", symbol, " after ", max_retries, " attempts.");
    return false;
}

int GetContextTimeframeShift(string symbol, datetime signal_bar_time)
{
   datetime context_bar_time = iTime(symbol, InpContextTimeframe, 0);
   if(signal_bar_time > context_bar_time)
   {
      return 1;
   }
   int shift = iBarShift(symbol, InpContextTimeframe, signal_bar_time, true);

   if(shift >= 0)
   {
      return shift;
   }
 
   Print("Warning: iBarShift failed for ", symbol, ". Using default shift.");
   return G_SignalCandleShift; // Fallback
}

int OnInit()
{
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(InpSlippage);
    trade.SetTypeFilling(ORDER_FILLING_FOK);

    int numSymbols = StringSplit(InpSymbolsToTrade, ',', G_SymbolsArray);
    if(numSymbols == 0) { Print("No symbols defined. EA exiting."); return(INIT_FAILED); }
    if(numSymbols > 64) { Print("Too many symbols (max 64). EA exiting."); return(INIT_FAILED); }

    ArrayResize(G_SymbolIndicatorHandles, numSymbols);
    ArrayResize(G_SymbolTradeStates, numSymbols);
    ArrayResize(G_SymbolCache, numSymbols);
    ArrayResize(G_WarmupComplete, numSymbols); // FIX #6

    for(int i = 0; i < numSymbols; i++)
    {
        string currentSymbol = G_SymbolsArray[i];
        StringTrimLeft(currentSymbol);
        StringTrimRight(currentSymbol);
        G_SymbolsArray[i] = currentSymbol;
        
        G_WarmupComplete[i] = false; 

        if(!SelectSymbolWithRetry(currentSymbol)) continue;
        
        trade.SetTypeFillingBySymbol(currentSymbol);
        CacheSymbolProperties(currentSymbol, i);

        G_SymbolIndicatorHandles[i].hSMA_Long_ContextTF = iMA(currentSymbol, InpContextTimeframe, InpSMA_Long_Period, 0, MODE_SMA, PRICE_CLOSE);
        G_SymbolIndicatorHandles[i].hADX_ContextTF = iADX(currentSymbol, InpContextTimeframe, InpADX_Period);
        G_SymbolIndicatorHandles[i].hATR_Volatility_ContextTF = iATR(currentSymbol, InpContextTimeframe, InpVolatility_ATR_Period);
        G_SymbolIndicatorHandles[i].hRSI_SignalTF = iRSI(currentSymbol, InpSignalTimeframe, InpRSI_Period, PRICE_CLOSE);
        G_SymbolIndicatorHandles[i].hSTOCH_SignalTF = iStochastic(currentSymbol, InpSignalTimeframe, InpSTOCH_K, InpSTOCH_D, InpSTOCH_Slowing, MODE_SMA, STO_LOWHIGH);
        G_SymbolIndicatorHandles[i].hATR_Management_SignalTF = iATR(currentSymbol, InpSignalTimeframe, InpSL_ATR_Period_SignalTF);

        if(G_SymbolIndicatorHandles[i].hSMA_Long_ContextTF == INVALID_HANDLE || G_SymbolIndicatorHandles[i].hADX_ContextTF == INVALID_HANDLE ||
           G_SymbolIndicatorHandles[i].hATR_Volatility_ContextTF == INVALID_HANDLE || G_SymbolIndicatorHandles[i].hRSI_SignalTF == INVALID_HANDLE ||
           G_SymbolIndicatorHandles[i].hSTOCH_SignalTF == INVALID_HANDLE || G_SymbolIndicatorHandles[i].hATR_Management_SignalTF == INVALID_HANDLE)
           {
               Print("Failed to initialize one or more indicators for ", currentSymbol, ". This symbol may not trade correctly.");
           }

        G_SymbolTradeStates[i].hasOpenTrade = false;
        G_SymbolTradeStates[i].ticket = 0;
    }

    ChartSetInteger(0, CHART_SHOW_TRADE_LEVELS, true);
    EventSetTimer(15);
    PrintFormat("%s initialized for %d symbols: %s", MQLInfoString(MQL_PROGRAM_NAME), numSymbols, InpSymbolsToTrade);
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    EventKillTimer();
    for(int i = 0; i < ArraySize(G_SymbolIndicatorHandles); i++)
    {
        if(G_SymbolIndicatorHandles[i].hSMA_Long_ContextTF != INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].hSMA_Long_ContextTF);
        if(G_SymbolIndicatorHandles[i].hADX_ContextTF != INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].hADX_ContextTF);
        if(G_SymbolIndicatorHandles[i].hATR_Volatility_ContextTF != INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].hATR_Volatility_ContextTF);
        if(G_SymbolIndicatorHandles[i].hRSI_SignalTF != INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].hRSI_SignalTF);
        if(G_SymbolIndicatorHandles[i].hSTOCH_SignalTF != INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].hSTOCH_SignalTF);
        if(G_SymbolIndicatorHandles[i].hATR_Management_SignalTF != INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].hATR_Management_SignalTF);
    }
    Print(MQLInfoString(MQL_PROGRAM_NAME) + " deinitialized. Reason: " + IntegerToString(reason));
}

void OnTick()
{
    if(!MQLInfoInteger(MQL_TRADE_ALLOWED) || !TerminalInfoInteger(TERMINAL_CONNECTED)) return;
    
    static datetime lastBarTime[];
    if(ArraySize(lastBarTime) != ArraySize(G_SymbolsArray))
    {
        ArrayResize(lastBarTime, ArraySize(G_SymbolsArray));
        for(int k=0; k<ArraySize(G_SymbolsArray); k++) lastBarTime[k]=0;
    }

    for(int i = 0; i < ArraySize(G_SymbolsArray); i++)
    {
        string currentSymbol = G_SymbolsArray[i];
        if(!SelectSymbolWithRetry(currentSymbol, 2)) continue;

        MqlRates ratesSignalTF[];
        if(CopyRates(currentSymbol, InpSignalTimeframe, 0, 1, ratesSignalTF) < 1) continue;

        if(ratesSignalTF[0].time > lastBarTime[i])
        {
            lastBarTime[i] = ratesSignalTF[0].time;
            ManageSymbol(currentSymbol, i);
        }
    }
}

void OnTimer()
{
    if(!MQLInfoInteger(MQL_TRADE_ALLOWED)) return;
    for(int i = 0; i < ArraySize(G_SymbolsArray); i++)
    {
        if(G_SymbolTradeStates[i].hasOpenTrade)
        {
            RefreshSymbolCache(G_SymbolsArray[i], i);
            ManageOpenPosition(G_SymbolsArray[i], G_SymbolTradeStates[i], i);
        }
    }
}

void ManageSymbol(string symbol, int symbol_idx)
{
    if(PositionSelect(symbol))
    {
        if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
        {
            if(!G_SymbolTradeStates[symbol_idx].hasOpenTrade)
            {
                G_SymbolTradeStates[symbol_idx].hasOpenTrade = true;
                G_SymbolTradeStates[symbol_idx].ticket = PositionGetInteger(POSITION_TICKET);
                G_SymbolTradeStates[symbol_idx].tradeType = (ENUM_ORDER_TYPE)PositionGetInteger(POSITION_TYPE);
                G_SymbolTradeStates[symbol_idx].openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
                G_SymbolTradeStates[symbol_idx].openTime = (datetime)PositionGetInteger(POSITION_TIME);
                G_SymbolTradeStates[symbol_idx].initialStopLossPrice = PositionGetDouble(POSITION_SL);
                G_SymbolTradeStates[symbol_idx].initialTakeProfitPrice = PositionGetDouble(POSITION_TP);
                G_SymbolTradeStates[symbol_idx].lotSize = PositionGetDouble(POSITION_VOLUME);
            }
            return;
        }
    }
    else if(G_SymbolTradeStates[symbol_idx].hasOpenTrade)
    {
        G_SymbolTradeStates[symbol_idx].hasOpenTrade = false;
    }

    if(InpUseTradingHours && !IsWithinTradingHours(InpStartHour, InpStartMinute, InpEndHour, InpEndMinute)) return;

    CheckEntryConditions(symbol, symbol_idx);
    UpdateChartComment(symbol, symbol_idx);
}

void CheckEntryConditions(string symbol, int symbol_idx)
{
    if(!CheckWarmupPeriod(symbol, symbol_idx)) return;

    int available_bars = iBars(symbol, InpSignalTimeframe);
    if(available_bars < G_SignalCandleShift + 2) return;
    
    int bars_to_copy = MathMin(available_bars, MathMax(InpDoji_Lookback_Bars, InpSwingPointLookback_SignalTF) + 5);
    MqlRates ratesSignalTF[];
    if(CopyRates(symbol, InpSignalTimeframe, 0, bars_to_copy, ratesSignalTF) < G_SignalCandleShift + 2) return;
    
    MqlRates signal_candle = ratesSignalTF[G_SignalCandleShift];
    int context_shift = GetContextTimeframeShift(symbol, signal_candle.time);

    double sma_long_val[], adx_val[], atr_volatility_val[], rsi_val[], stoch_main[], stoch_signal[], atr_sl_val[];
    
    if(CopyBuffer(G_SymbolIndicatorHandles[symbol_idx].hSMA_Long_ContextTF, 0, context_shift, 1, sma_long_val) != 1 ||
       CopyBuffer(G_SymbolIndicatorHandles[symbol_idx].hRSI_SignalTF, 0, G_SignalCandleShift, 1, rsi_val) != 1 ||
       CopyBuffer(G_SymbolIndicatorHandles[symbol_idx].hATR_Management_SignalTF, 0, G_SignalCandleShift, 1, atr_sl_val) != 1)
    {
        Print(symbol, ": Critical indicators (SMA/RSI/ATR_SL) failed. Skipping check.");
        return;
    }
    
    if(CopyBuffer(G_SymbolIndicatorHandles[symbol_idx].hADX_ContextTF, MAIN_LINE, context_shift, 1, adx_val) != 1) { ArrayResize(adx_val,1); adx_val[0] = 0; } // Neutral ADX
    if(CopyBuffer(G_SymbolIndicatorHandles[symbol_idx].hATR_Volatility_ContextTF, 0, context_shift, 1, atr_volatility_val) != 1) { ArrayResize(atr_volatility_val,1); atr_volatility_val[0] = 99999; } // High ATR to allow trade
    if(CopyBuffer(G_SymbolIndicatorHandles[symbol_idx].hSTOCH_SignalTF, MAIN_LINE, G_SignalCandleShift, 2, stoch_main) != 2 ||
       CopyBuffer(G_SymbolIndicatorHandles[symbol_idx].hSTOCH_SignalTF, SIGNAL_LINE, G_SignalCandleShift, 2, stoch_signal) != 2)
    {
        ArrayResize(stoch_main,2); ArrayResize(stoch_signal,2); 
        stoch_main[0]=50; stoch_main[1]=50; stoch_signal[0]=50; stoch_signal[1]=50; // Neutral Stochastic
    }

    double point_val = G_SymbolCache[symbol_idx].point;
    if (atr_volatility_val[0] < (InpVolatility_Min_ATR_Pips_ContextTF * GetPipValueMultiplier(symbol, point_val))) return;
    if (adx_val[0] >= InpADX_Max_Level) return;

    int doji_count = 0;
    for(int k=G_SignalCandleShift; k < G_SignalCandleShift + MathMin(InpDoji_Lookback_Bars, ArraySize(ratesSignalTF) - G_SignalCandleShift); k++)
    {
        if(IsDoji(ratesSignalTF[k], point_val)) doji_count++;
    }
    if(doji_count > InpDoji_Max_Allowed) return;

    ENUM_ORDER_TYPE order_type = WRONG_VALUE;
    double entry_price = 0;

    bool stoch_long_ok = (stoch_main[0] >= InpSTOCH_Long_Min_Level_Main) || (stoch_main[1] < InpSTOCH_Long_Cross_From_Oversold && stoch_main[0] > InpSTOCH_Long_Cross_From_Oversold && stoch_main[0] > stoch_signal[0] && stoch_main[1] <= stoch_signal[1]);
    double candle_range_long = signal_candle.high - signal_candle.low;
    if(candle_range_long < point_val) candle_range_long = point_val;
    double long_close_pos_ratio = (signal_candle.close - signal_candle.low) / candle_range_long;

    bool long_signal = signal_candle.close > sma_long_val[0] &&
                       rsi_val[0] > InpRSI_Long_Min_Level &&
                       stoch_long_ok &&
                       long_close_pos_ratio >= InpClosePosRatio_Long_Min &&
                       (!InpUsePinBarFilter || IsBullishPinBar(signal_candle, point_val));

    if(long_signal)
    {
        order_type = ORDER_TYPE_BUY;
        entry_price = SymbolInfoDouble(symbol, SYMBOL_ASK);
    }
    else 
    {
        bool stoch_short_ok = (stoch_main[0] <= InpSTOCH_Short_Max_Level_Main) || (stoch_main[1] > InpSTOCH_Short_Cross_From_Overbought && stoch_main[0] < InpSTOCH_Short_Cross_From_Overbought && stoch_main[0] < stoch_signal[0] && stoch_main[1] >= stoch_signal[1]);
        double candle_range_short = signal_candle.high - signal_candle.low;
        if(candle_range_short < point_val) candle_range_short = point_val;
        double short_close_pos_ratio = (signal_candle.close - signal_candle.low) / candle_range_short;
        
        bool short_signal = signal_candle.close < sma_long_val[0] &&
                            rsi_val[0] < InpRSI_Short_Max_Level &&
                            stoch_short_ok &&
                            short_close_pos_ratio <= InpClosePosRatio_Short_Max &&
                            (!InpUsePinBarFilter || IsBearishPinBar(signal_candle, point_val));
        
        if(short_signal)
        {
            order_type = ORDER_TYPE_SELL;
            entry_price = SymbolInfoDouble(symbol, SYMBOL_BID);
        }
    }

    if(order_type == WRONG_VALUE) return;

    double sl_price = CalculateStopLossPrice(symbol, order_type, signal_candle, ratesSignalTF, atr_sl_val[0], point_val);
    if(sl_price == 0)
    {
        double fallback_sl_dist = 50 * GetPipValueMultiplier(symbol, point_val);
        sl_price = (order_type == ORDER_TYPE_BUY) ? (entry_price - fallback_sl_dist) : (entry_price + fallback_sl_dist);
        Print(symbol, ": SL calculation failed, using fixed distance fallback.");
    }
    
    double stop_loss_distance_price = MathAbs(entry_price - sl_price);
    if (stop_loss_distance_price < G_SymbolCache[symbol_idx].stopLevel * point_val)
    {
         Print(symbol, ": Calculated SL is too close to the market price. Signal rejected.");
         return;
    }
    
    double tp_price = 0;
    if(order_type == ORDER_TYPE_BUY)
        tp_price = entry_price + (stop_loss_distance_price * InpMinRiskReward);
    else
        tp_price = entry_price - (stop_loss_distance_price * InpMinRiskReward);
    
    double stop_loss_points_for_lot = stop_loss_distance_price / point_val;
    
    double lot_size = CalculateLotSize(symbol, stop_loss_points_for_lot, symbol_idx);
    if(lot_size <= 0)
    {
        lot_size = G_SymbolCache[symbol_idx].minLot;
        Print(symbol, ": Lot calculation failed, falling back to minimum lot size: ", DoubleToString(lot_size, 2));
    }
    if(lot_size < G_SymbolCache[symbol_idx].minLot) return; // Still can't trade if min lot is invalid
    
    ulong ticket_nr = 0;
    string comment = MQLInfoString(MQL_PROGRAM_NAME) + (order_type == ORDER_TYPE_BUY ? " Buy" : " Sell");
    OpenPosition(symbol, order_type, lot_size, entry_price, sl_price, tp_price, comment, ticket_nr, symbol_idx);
}

void  ManageOpenPosition(string symbol, TradeState& trade_state_ref, int symbol_idx)
{
    if(!trade_state_ref.hasOpenTrade || !PositionSelectByTicket(trade_state_ref.ticket))
    {
        trade_state_ref.hasOpenTrade = false;
        return;
    }

    ulong current_ticket = PositionGetInteger(POSITION_TICKET);
    if(current_ticket != trade_state_ref.ticket)
    {
        trade_state_ref.hasOpenTrade = false;
        return;
    }

    double current_sl = PositionGetDouble(POSITION_SL);
    double current_tp = PositionGetDouble(POSITION_TP);
    double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
    ENUM_ORDER_TYPE order_type = (ENUM_ORDER_TYPE)PositionGetInteger(POSITION_TYPE);
    datetime ticket_open_time = (datetime)PositionGetInteger(POSITION_TIME);
    double current_bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double current_ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    int digits = G_SymbolCache[symbol_idx].digits;

    MqlRates ratesSignalTF_Manage[];
    if(CopyRates(symbol, InpSignalTimeframe, 0, 1, ratesSignalTF_Manage) < 1) return;
    long bars_held = (ratesSignalTF_Manage[0].time - ticket_open_time) / PeriodSeconds(InpSignalTimeframe);
    if(bars_held >= InpMaxHoldTime_Bars_SignalTF)
    {
        Print(symbol, " Trade ", trade_state_ref.ticket, " hit max hold time (", IntegerToString(bars_held), " bars). Closing.");
        if(trade.PositionClose(trade_state_ref.ticket, InpSlippage))
        {
            trade_state_ref.hasOpenTrade = false;
        }
        else Print("Failed to close position by time: ", trade.ResultRetcodeDescription());
        return;
    }

    double atr_manage_val[1];
    if(CopyBuffer(G_SymbolIndicatorHandles[symbol_idx].hATR_Management_SignalTF, 0, 0, 1, atr_manage_val) < 1)
    {
        Print(symbol, " Failed to get ATR for trade management. Skipping BE/Trail.");
        return;
    }
    if(atr_manage_val[0] <= 0) return;

    if(InpUseBreakeven)
    {
        double profit_in_price = (order_type == ORDER_TYPE_BUY) ? (current_bid - open_price) : (open_price - current_ask);
        double be_trigger_profit_price = atr_manage_val[0] * InpBreakeven_ATR_Multiplier_Profit;
        if (profit_in_price >= be_trigger_profit_price)
        {
            bool modify_to_be = (order_type == ORDER_TYPE_BUY && current_sl < open_price) || (order_type == ORDER_TYPE_SELL && current_sl > open_price);
            if(modify_to_be)
            {
                if(trade.PositionModify(trade_state_ref.ticket, open_price, current_tp))
                {
                    Print(symbol, " Trade ", trade_state_ref.ticket, " SL moved to Breakeven: ", DoubleToString(open_price, digits));
                }
                else Print("Failed to modify for BE: ", trade.ResultRetcodeDescription());
            }
        }
    }

    if(InpUseTrailingStop)
    {
        double new_sl_price = 0;
        double trail_distance_price = atr_manage_val[0] * InpTrail_ATR_Multiplier;
        if(order_type == ORDER_TYPE_BUY)
        {
            new_sl_price = current_bid - trail_distance_price;
            if(new_sl_price > current_sl && new_sl_price > open_price)
            {
                if(!trade.PositionModify(trade_state_ref.ticket, new_sl_price, current_tp))
                {
                   Print("Failed to modify for Trail SL Buy: ", trade.ResultRetcodeDescription());
                }
            }
        }
        else
        {
            new_sl_price = current_ask + trail_distance_price;
            if(new_sl_price < current_sl && new_sl_price < open_price)
            {
                if(!trade.PositionModify(trade_state_ref.ticket, new_sl_price, current_tp))
                {
                    Print("Failed to modify for Trail SL Sell: ", trade.ResultRetcodeDescription());
                }
            }
        }
    }
    UpdateChartComment(symbol, symbol_idx);
}

// ... [All other helper functions like CalculateLotSize, CalculateStopLossPrice, et

bool OpenPosition(string symbol, ENUM_ORDER_TYPE order_type, double lot_size,
                  double& current_market_price_ref,
                  double& sl_price_ref,
                  double& tp_price_ref,
                  string comment, ulong& ticket_nr_ref, int symbol_idx)
{
    double current_ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double current_bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    
    current_market_price_ref = (order_type == ORDER_TYPE_BUY) ? current_ask : current_bid;
    
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    double min_stop_level_points = (double)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
    double min_stop_dist_price = (min_stop_level_points == 0 ? 2*point : min_stop_level_points * point);

    if(order_type == ORDER_TYPE_BUY)
    {
        if(sl_price_ref == 0 || sl_price_ref >= current_market_price_ref - min_stop_dist_price)
            sl_price_ref = current_market_price_ref - (min_stop_dist_price + 2 * point);
        if(tp_price_ref != 0 && tp_price_ref <= current_market_price_ref + min_stop_dist_price)
            tp_price_ref = current_market_price_ref + (min_stop_dist_price + 2 * point);
    }
    else 
    {
        if(sl_price_ref == 0 || sl_price_ref <= current_market_price_ref + min_stop_dist_price)
            sl_price_ref = current_market_price_ref + (min_stop_dist_price + 2 * point);
        if(tp_price_ref != 0 && tp_price_ref >= current_market_price_ref - min_stop_dist_price)
            tp_price_ref = current_market_price_ref - (min_stop_dist_price + 2 * point);
    }

    sl_price_ref = NormalizeDouble(sl_price_ref, digits);
    tp_price_ref = (tp_price_ref == current_market_price_ref || tp_price_ref == 0) ? 0 : NormalizeDouble(tp_price_ref, digits);

    bool trade_result = false;
    if(order_type == ORDER_TYPE_BUY)
    {
        trade_result = trade.Buy(lot_size, symbol, current_ask, sl_price_ref, tp_price_ref, comment);
    }
    else if(order_type == ORDER_TYPE_SELL)
    {
        trade_result = trade.Sell(lot_size, symbol, current_bid, sl_price_ref, tp_price_ref, comment);
    }

    if(trade_result)
    {
        ticket_nr_ref = trade.ResultOrder();
        
        G_SymbolTradeStates[symbol_idx].hasOpenTrade = true;
        G_SymbolTradeStates[symbol_idx].ticket = ticket_nr_ref;
        G_SymbolTradeStates[symbol_idx].tradeType = order_type;
        G_SymbolTradeStates[symbol_idx].openPrice = trade.ResultPrice(); // CTrade provides actual fill price
        G_SymbolTradeStates[symbol_idx].openTime = TimeCurrent();
        G_SymbolTradeStates[symbol_idx].initialStopLossPrice = sl_price_ref;
        G_SymbolTradeStates[symbol_idx].initialTakeProfitPrice = tp_price_ref;
        G_SymbolTradeStates[symbol_idx].lotSize = lot_size;

        Print(symbol, ": Position opened. Ticket:", ticket_nr_ref, ", Type:", EnumToString(order_type),
              ", Lots:", DoubleToString(lot_size,2), ", Entry:", DoubleToString(G_SymbolTradeStates[symbol_idx].openPrice, digits),
              ", SL:", DoubleToString(sl_price_ref, digits), ", TP:", DoubleToString(tp_price_ref, digits));
        return true;
    }
    else
    {
        Print("Trade execution failed for ", symbol, ". Error: ", trade.ResultRetcode(), " (", trade.ResultRetcodeDescription(), ")");
        return false;
    }
}

double CalculateLotSize(string symbol, double stop_loss_points_val, int symbol_idx)
{
    if(stop_loss_points_val <= 0) return 0.0;
    
    double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double risk_amount = account_equity * (InpRiskPercentPerTrade / 100.0);
    if(risk_amount <= 0) return 0.0;

    double contract_size = G_SymbolCache[symbol_idx].contractSize;
    double tick_value = G_SymbolCache[symbol_idx].tickValue;
    double tick_size = G_SymbolCache[symbol_idx].tickSize;
    double point = G_SymbolCache[symbol_idx].point;

    if(tick_value <= 0 || tick_size <= 0 || contract_size <= 0 || point <= 0)
    {
        Print(symbol, ": Invalid cached symbol properties for lot calculation.");
        return 0.0;
    }
    
    double value_per_point_per_lot = tick_value / (tick_size / point);

    if(G_SymbolCache[symbol_idx].profitCurrency != AccountInfoString(ACCOUNT_CURRENCY))
    {
        string conversion_pair_to_acc = G_SymbolCache[symbol_idx].profitCurrency + AccountInfoString(ACCOUNT_CURRENCY);
        string conversion_pair_from_acc = AccountInfoString(ACCOUNT_CURRENCY) + G_SymbolCache[symbol_idx].profitCurrency;
        double rate_to_acc = SymbolInfoDouble(conversion_pair_to_acc, SYMBOL_BID);
        if(rate_to_acc == 0) rate_to_acc = 1.0 / SymbolInfoDouble(conversion_pair_from_acc, SYMBOL_ASK);
        if(rate_to_acc == 0) { Print("Cannot find conversion rate for ", conversion_pair_to_acc); return 0.0;}
        value_per_point_per_lot *= rate_to_acc;
    }

    if(value_per_point_per_lot <= 1e-9) return 0.0;

    double loss_per_lot = stop_loss_points_val * value_per_point_per_lot;
    if(loss_per_lot <= 1e-9) return 0.0;

    double desired_lot_size = risk_amount / loss_per_lot;
    
    desired_lot_size = MathFloor(desired_lot_size / G_SymbolCache[symbol_idx].lotStep) * G_SymbolCache[symbol_idx].lotStep;
    desired_lot_size = MathMax(G_SymbolCache[symbol_idx].minLot, desired_lot_size);
    desired_lot_size = MathMin(G_SymbolCache[symbol_idx].maxLot, desired_lot_size);
    
    if(desired_lot_size < G_SymbolCache[symbol_idx].minLot && G_SymbolCache[symbol_idx].minLot > 0) 
        desired_lot_size = 0.0;

    return NormalizeDouble(desired_lot_size, 2);
}

double CalculateStopLossPrice(string symbol, ENUM_ORDER_TYPE order_type, const MqlRates &signal_bar, const MqlRates &history_rates[], double atr_value, double point)

{
    double sl_price = 0;

    int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    if(InpUseSwingPointSL)
    {
        if(order_type == ORDER_TYPE_BUY)
        {
            double swing_low = signal_bar.low;
            for(int k = G_SignalCandleShift + 1; k < G_SignalCandleShift + 1 + InpSwingPointLookback_SignalTF; k++)
            {
                if(k < ArraySize(history_rates)) // Ensure index is within bounds
                    swing_low = MathMin(swing_low, history_rates[k].low);
                else break;
            }
            sl_price = swing_low - atr_value * 0.2; 
        }
        else 
        {
            double swing_high = signal_bar.high;
            for(int k = G_SignalCandleShift + 1; k < G_SignalCandleShift + 1 + InpSwingPointLookback_SignalTF; k++)
            {
                if(k < ArraySize(history_rates))
                    swing_high = MathMax(swing_high, history_rates[k].high);
                else break;
            }
            sl_price = swing_high + atr_value * 0.2; 
        }
    }

    else

    {
        double reference_price = signal_bar.close;
        if(order_type == ORDER_TYPE_BUY)
            sl_price = reference_price - (atr_value * InpSL_ATR_Multiplier);
        else 

            sl_price = reference_price + (atr_value * InpSL_ATR_Multiplier);

    }

    double current_price_for_sl_check = (order_type == ORDER_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_ASK) : SymbolInfoDouble(symbol, SYMBOL_BID);

    double min_stop_dist_price = (double)SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;

    if(order_type == ORDER_TYPE_BUY)

    {

        if(current_price_for_sl_check - sl_price < min_stop_dist_price)

            sl_price = current_price_for_sl_check - (min_stop_dist_price + 2 * point); // Add a small buffer

    }
    else
    {

        if(sl_price - current_price_for_sl_check < min_stop_dist_price)

            sl_price = current_price_for_sl_check + (min_stop_dist_price + 2 * point);

    }

    return NormalizeDouble(sl_price, digits);
}

void UpdateChartComment(string symbol, int symbol_idx)
{
    if(symbol != _Symbol) return;
    
    string comment = "--- Code 2 EA Status ---\n";
    comment += "Symbol: " + symbol + "\n";
    comment += "Trade Allowed: " + (MQLInfoInteger(MQL_TRADE_ALLOWED) ? "Yes" : "No") + "\n";
    
    if(G_SymbolTradeStates[symbol_idx].hasOpenTrade)
    {
        if(PositionSelect(symbol) && PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
        {
            string p_type = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) ? "BUY" : "SELL";
            comment += "Status: Position Open\n";
            comment += "Ticket: " + (string)G_SymbolTradeStates[symbol_idx].ticket + "\n";
            comment += "Type: " + p_type + "\n";
            comment += "Profit: " + DoubleToString(PositionGetDouble(POSITION_PROFIT), 2) + "\n";
        }
        else
        {
            comment += "Status: Position tracking error\n";
        }
    }
    else
    {
        comment += "Status: Monitoring for signal...\n";
    }    
    Comment(comment);
}

bool IsWithinTradingHours(int start_hour, int start_minute, int end_hour, int end_minute)

{

    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int current_total_minutes = dt.hour * 60 + dt.min;
    int start_total_minutes = start_hour * 60 + start_minute;
    int end_total_minutes = end_hour * 60 + end_minute;
    if(start_total_minutes <= end_total_minutes) 
        return (current_total_minutes >= start_total_minutes && current_total_minutes <= end_total_minutes);
    else 
        return (current_total_minutes >= start_total_minutes || current_total_minutes <= end_total_minutes);
}

bool IsDoji(const MqlRates &candle, double point)

{
    double body = MathAbs(candle.open - candle.close);
    double range = candle.high - candle.low;
    if(range < point * 5) return false;
    return (body / range < 0.1); 

}

bool IsBullishPinBar(const MqlRates &candle, double point)
{
    double body = MathAbs(candle.open - candle.close);
    double range = candle.high - candle.low;
    if(range < point * 10) return false; 
    double lower_wick = MathMin(candle.open, candle.close) - candle.low;
    double upper_wick = candle.high - MathMax(candle.open, candle.close);
    return (lower_wick > body * 2.0 && upper_wick < body * 0.75 && body / range < 0.33 && candle.close > candle.open);
}

bool IsBearishPinBar(const MqlRates &candle, double point)
{
    double body = MathAbs(candle.open - candle.close);
    double range = candle.high - candle.low;
    if(range < point * 10) return false;
    double lower_wick = MathMin(candle.open, candle.close) - candle.low;
    double upper_wick = candle.high - MathMax(candle.open, candle.close);
    return (upper_wick > body * 2.0 && lower_wick < body * 0.75 && body / range < 0.33 && candle.close < candle.open);
}

double GetPipValueMultiplier(string symbol, double point_value)

{
    int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
    if (StringFind(symbol, "JPY") != -1 && digits == 3) return point_value * 100; 
    if ((digits == 5 || digits == 3) && StringFind(symbol, "JPY") == -1) return point_value * 10; 
    if (digits == 2 && (StringFind(symbol,"XAU")!=-1 || StringFind(symbol,"GOLD")!=-1)) return point_value *10;
    return point_value; 
}