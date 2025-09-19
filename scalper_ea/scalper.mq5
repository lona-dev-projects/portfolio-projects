
#include <Trade\Trade.mqh> // CTrade class for trading operations [1, 2, 3]

CTrade trade;

input group           "General Settings"
input string          InpSymbolsToTrade = "EURUSD,GBPUSD,USDJPY,XAUUSD"; 
input ulong           InpMagicNumber = 67890;                  
input int             InpSlippage = 5;                            
input double          InpRiskPercentPerTrade = 13.0;          
input ENUM_TIMEFRAMES InpSignalTimeframe = PERIOD_M1;           
input ENUM_TIMEFRAMES InpTrendTimeframe = PERIOD_M5;             

input group           "Trading Hours (Server Time)"
input int             InpBreakoutStartTimeH = 9;                 
input int             InpBreakoutStartTimeM = 45;                
input int             InpBreakoutEndTimeH = 11;                   
input int             InpBreakoutEndTimeM = 30;                   
input bool            InpUsePullbackTimeWindow = false;           
input int             InpPullbackStartTimeH = 0;
input int             InpPullbackStartTimeM = 0;
input int             InpPullbackEndTimeH = 23;
input int             InpPullbackEndTimeM = 59;

input group           "Pullback Strategy Parameters"
input double          InpVWAP_Proximity_Percent = 0.1;         
input double          InpPullbackEntryVolumeFactor = 1.4;        
input double          InpPullbackTakeProfit1_Percent = 0.5;      
input double          InpPullbackTakeProfit2_Percent = 0.75;      
input double          InpPullbackStopLoss_Percent = 0.3;          
input int             InpPullbackMaxHoldTime_Minutes = 3;         
input int             InpPullbackExitIfNotHit1Min_Minutes = 1;    
// Breakout Strategy Parameters [18]
input group           "Breakout Strategy Parameters"
input double          InpBreakoutSqueezeRange_Percent = 0.1;      
input int             InpBreakoutSqueezeBars = 3;                 
input double          InpBreakoutSqueezeVolumeFactor = 0.8;       
input double          InpBreakoutEntryVolumeFactor = 1.5;         
input double          InpBreakoutConfirmationVolumeFactor = 1.2;  
// input double       InpBreakoutOIRisePercent = 5.0;             
input double          InpBreakoutCandleWickMaxPercent = 0.1;      
input int             InpBreakoutRetestMaxBars = 3;               
input double          InpBreakoutTakeProfit_Percent = 0.5;        
input double          InpBreakoutStopLoss_Percent = 0.2;         
input int             InpBreakoutExitIfNotHit_Minutes = 1;       

input group           "Indicator Settings"
input int             InpFastEMA_Period = 5;                      
input int             InpSlowEMA_Period = 13;                     
input int             InpTrendEMA_Period = 50;                    
input int             InpVolumeMA_Period = 15;                   
input int             InpATR_Period = 14;                         
input int             InpRSI_Period = 7;                          
input int             InpRSI_Overbought = 70;                     
input int             InpRSI_Oversold = 30;                       
input int             InpMACD_FastEMA = 3;                        
input int             InpMACD_SlowEMA = 10;                       
input int             InpMACD_SignalSMA = 16;                     


string G_SymbolsArray[]; 
int    G_SignalCandle = 1; 
struct SymbolIndicators
{
    int h_FastEMA_SignalTF;
    int h_SlowEMA_SignalTF;
    int h_TrendEMA_TrendTF;
    int h_ATR_SignalTF;
    int h_RSI_TrendTF;
    int h_MACD_TrendTF;
    // VWAP is calculated, not a standard handle. Volume is also calculated from MqlRates or iVolume.
};
SymbolIndicators G_SymbolIndicatorHandles[];

enum StrategyType { STRATEGY_NONE, STRATEGY_PULLBACK, STRATEGY_BREAKOUT };
struct TradeState
{
    bool              hasOpenTrade;
    ulong             ticket;
    ENUM_ORDER_TYPE   tradeType;     
    double            openPrice;
    double            initialStopLossPrice;
    double            initialTakeProfitPrice; 
    datetime          openTime;
    bool              partialTp1Hit
    StrategyType      strategyInvolved;
    double            lotSize;       
    int               breakoutBarIndex; 
    datetime          breakoutOccurredTime; 
};
TradeState G_SymbolTradeStates[];
struct DailyVWAP
{
    double   sum_price_volume;
    long     sum_volume;
    datetime current_day_start_time; 
};
DailyVWAP G_DailyVWAP_Data[]; // One per symbol

enum ENUM_TREND_STATE { TREND_NONE_SIDEWAYS, TREND_UP, TREND_DOWN };

int OnInit()
{

    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(InpSlippage);
    trade.SetTypeFilling(ORDER_FILLING_FOK); // Fill Or Kill: good for scalping to avoid partial fills at bad prices [51]
    
    int numSymbols = StringSplit(InpSymbolsToTrade, ',', G_SymbolsArray);
    if(numSymbols == 0)
    {
        Print("No symbols specified in InpSymbolsToTrade. EA will not run.");
        return(INIT_FAILED);
    }
    if(numSymbols > 64) // Practical limit for arrays/performance
    {
        Print("Too many symbols specified. Max 64. EA will not run.");
        return(INIT_FAILED);
    }

    ArrayResize(G_SymbolIndicatorHandles, numSymbols);
    ArrayResize(G_SymbolTradeStates, numSymbols);
    ArrayResize(G_DailyVWAP_Data, numSymbols);


    for(int i = 0; i < numSymbols; i++)
    {
        string currentSymbol = G_SymbolsArray[i];
        StringTrimLeft(currentSymbol); StringTrimRight(currentSymbol); // Clean up spaces
        G_SymbolsArray[i] = currentSymbol; // Store cleaned symbol name

        if(!SymbolSelect(currentSymbol, true))
        {
            Print("Failed to select symbol: ", currentSymbol, " in MarketWatch. Please add it. Error: ", IntegerToString(GetLastError()));
            // Consider this a critical failure for the specific symbol or entire EA
            // For now, iI will let it try to initialize indicators, but it might fail.
        }
        trade.SetTypeFillingBySymbol(currentSymbol); // Set filling mode per symbol

        G_SymbolIndicatorHandles[i].h_FastEMA_SignalTF = iMA(currentSymbol, InpSignalTimeframe, InpFastEMA_Period, 0, MODE_EMA, PRICE_CLOSE);
        if(G_SymbolIndicatorHandles[i].h_FastEMA_SignalTF == INVALID_HANDLE) { Print("Failed to create Fast EMA handle for " + currentSymbol + ", TF: " + EnumToString(InpSignalTimeframe) + ", Error: " + IntegerToString(_LastError)); return(INIT_FAILED); }

        G_SymbolIndicatorHandles[i].h_SlowEMA_SignalTF = iMA(currentSymbol, InpSignalTimeframe, InpSlowEMA_Period, 0, MODE_EMA, PRICE_CLOSE);
        if(G_SymbolIndicatorHandles[i].h_SlowEMA_SignalTF == INVALID_HANDLE) { Print("Failed to create Slow EMA handle for " + currentSymbol + ", TF: " + EnumToString(InpSignalTimeframe) + ", Error: " + IntegerToString(_LastError)); return(INIT_FAILED); }

        G_SymbolIndicatorHandles[i].h_TrendEMA_TrendTF = iMA(currentSymbol, InpTrendTimeframe, InpTrendEMA_Period, 0, MODE_EMA, PRICE_CLOSE);
        if(G_SymbolIndicatorHandles[i].h_TrendEMA_TrendTF == INVALID_HANDLE) { Print("Failed to create Trend EMA handle for " + currentSymbol + ", TF: " + EnumToString(InpTrendTimeframe) + ", Error: " + IntegerToString(_LastError)); return(INIT_FAILED); }

        G_SymbolIndicatorHandles[i].h_ATR_SignalTF = iATR(currentSymbol, InpSignalTimeframe, InpATR_Period);
        if(G_SymbolIndicatorHandles[i].h_ATR_SignalTF == INVALID_HANDLE) { Print("Failed to create ATR handle for " + currentSymbol + ", TF: " + EnumToString(InpSignalTimeframe) + ", Error: " + IntegerToString(_LastError)); return(INIT_FAILED); }

        G_SymbolIndicatorHandles[i].h_RSI_TrendTF = iRSI(currentSymbol, InpTrendTimeframe, InpRSI_Period, PRICE_CLOSE);
        if(G_SymbolIndicatorHandles[i].h_RSI_TrendTF == INVALID_HANDLE) { Print("Failed to create RSI handle for " + currentSymbol + ", TF: " + EnumToString(InpTrendTimeframe) + ", Error: " + IntegerToString(_LastError)); return(INIT_FAILED); }

        G_SymbolIndicatorHandles[i].h_MACD_TrendTF = iMACD(currentSymbol, InpTrendTimeframe, InpMACD_FastEMA, InpMACD_SlowEMA, InpMACD_SignalSMA, PRICE_CLOSE);
        if(G_SymbolIndicatorHandles[i].h_MACD_TrendTF == INVALID_HANDLE) { Print("Failed to create MACD handle for " + currentSymbol + ", TF: " + EnumToString(InpTrendTimeframe) + ", Error: " + IntegerToString(_LastError)); return(INIT_FAILED); }

        G_SymbolTradeStates[i].hasOpenTrade = false;
        G_SymbolTradeStates[i].ticket = 0;
        G_SymbolTradeStates[i].partialTp1Hit = false;
        G_SymbolTradeStates[i].strategyInvolved = STRATEGY_NONE;
        G_SymbolTradeStates[i].breakoutBarIndex = -1;
        G_SymbolTradeStates[i].breakoutOccurredTime = 0;

        G_DailyVWAP_Data[i].sum_price_volume = 0;
        G_DailyVWAP_Data[i].sum_volume = 0;
        G_DailyVWAP_Data[i].current_day_start_time = 0;


        MqlRates rates_dummy[];
        int bars_needed_signal = MathMax(InpFastEMA_Period, InpSlowEMA_Period) + InpATR_Period + InpVolumeMA_Period + 5; // Max lookback for signal TF
        if(CopyRates(currentSymbol, InpSignalTimeframe, 0, bars_needed_signal, rates_dummy) < bars_needed_signal)
           Print("Warning: Insufficient history for " + currentSymbol + " on " + EnumToString(InpSignalTimeframe) + " for initial calculations. Need " + IntegerToString(bars_needed_signal) + " bars.");
        
        int bars_needed_trend = MathMax(InpTrendEMA_Period, MathMax(InpRSI_Period, InpMACD_SlowEMA + InpMACD_SignalSMA)) + 5; // Max lookback for trend TF
        if(CopyRates(currentSymbol, InpTrendTimeframe, 0, bars_needed_trend, rates_dummy) < bars_needed_trend)
           Print("Warning: Insufficient history for " + currentSymbol + " on " + EnumToString(InpTrendTimeframe) + " for trend indicators. Need " + IntegerToString(bars_needed_trend) + " bars.");
    }

    
    ChartSetInteger(0, CHART_SHOW_TRADE_LEVELS, true);
    EventSetTimer(1); // Set a timer for 1 second for periodic checks (e.g. time-based exits)
    //---
    PrintFormat("UltimateScalperEA initialized successfully for %d symbols: %s", numSymbols, InpSymbolsToTrade);
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
    EventKillTimer();
    for(int i = 0; i < ArraySize(G_SymbolIndicatorHandles); i++)
    {
        if(G_SymbolIndicatorHandles[i].h_FastEMA_SignalTF!= INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].h_FastEMA_SignalTF);
        if(G_SymbolIndicatorHandles[i].h_SlowEMA_SignalTF!= INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].h_SlowEMA_SignalTF);
        if(G_SymbolIndicatorHandles[i].h_TrendEMA_TrendTF!= INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].h_TrendEMA_TrendTF);
        if(G_SymbolIndicatorHandles[i].h_ATR_SignalTF!= INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].h_ATR_SignalTF);
        if(G_SymbolIndicatorHandles[i].h_RSI_TrendTF!= INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].h_RSI_TrendTF);
        if(G_SymbolIndicatorHandles[i].h_MACD_TrendTF!= INVALID_HANDLE) IndicatorRelease(G_SymbolIndicatorHandles[i].h_MACD_TrendTF);
    }
    Print("UltimateScalperEA deinitialized. Reason: ", reason);
    //I have to consider option to close all trades on deinit
}

void OnTick()
{
    if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
    {
       return; // Trading not allowed
    }
   
    //Iterate Through Symbols
    for(int i = 0; i < ArraySize(G_SymbolsArray); i++)
    {
        string currentSymbol = G_SymbolsArray[i];
        // Ensure symbol is available in MarketWatch and data is current
        if(SymbolInfoInteger(currentSymbol, SYMBOL_SELECT) == false)
        {
            if(!SymbolSelect(currentSymbol,true))
            {
                // Print("Symbol ", currentSymbol, " is not available in MarketWatch. Skipping.");
                continue;
            }
        }
        datetime last_tick_time = (datetime)SymbolInfoInteger(currentSymbol, SYMBOL_TIME);
        if(TimeCurrent() - last_tick_time > 300) // No ticks for 5 minutes, market might be closed or stale
        {
            // Print("Symbol ", currentSymbol, " has stale data. Skipping.");
            continue;
        }

        ManageSymbol(currentSymbol, i);
    }
}

void OnTimer()
{
    // Iterate through symbols and check G_SymbolTradeStates for time-based exits
    for(int i = 0; i < ArraySize(G_SymbolsArray); i++)
    {
        if(G_SymbolTradeStates[i].hasOpenTrade)
        {
    
            string currentSymbol = G_SymbolsArray[i];
            if(SymbolInfoInteger(currentSymbol, SYMBOL_SELECT) == false) continue;
            datetime last_tick_time = (datetime)SymbolInfoInteger(currentSymbol, SYMBOL_TIME);
            if(TimeCurrent() - last_tick_time > 300) continue;

            ManageOpenPosition(currentSymbol, G_SymbolTradeStates[i], i);
        }
    }
}

void ManageSymbol(string symbol, int symbol_idx)
{
    if(PositionSelect(symbol)) // A position exists for this symbol
    {
        if(PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
        {
            // Our EA opened this position. Update state and manage it.
            if(!G_SymbolTradeStates[symbol_idx].hasOpenTrade) // If state was somehow lost, re-populate
            {
                G_SymbolTradeStates[symbol_idx].hasOpenTrade = true;
                G_SymbolTradeStates[symbol_idx].ticket = PositionGetInteger(POSITION_TICKET);
                G_SymbolTradeStates[symbol_idx].tradeType = (ENUM_ORDER_TYPE)PositionGetInteger(POSITION_TYPE);
                G_SymbolTradeStates[symbol_idx].openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
                G_SymbolTradeStates[symbol_idx].openTime = (datetime)PositionGetInteger(POSITION_TIME);
                G_SymbolTradeStates[symbol_idx].initialStopLossPrice = PositionGetDouble(POSITION_SL);
                G_SymbolTradeStates[symbol_idx].initialTakeProfitPrice = PositionGetDouble(POSITION_TP);
                G_SymbolTradeStates[symbol_idx].lotSize = PositionGetDouble(POSITION_VOLUME);
                // strategyInvolved and partialTp1Hit should persist if EA restarted, but may need loading from history/comment
                // For simplicity, if EA restarts, it might lose this specific state for ongoing trades unless I save it.
            }
            ManageOpenPosition(symbol, G_SymbolTradeStates[symbol_idx], symbol_idx);
            return; // Only one trade per symbol
        }
    }
    else 
    {
        if(G_SymbolTradeStates[symbol_idx].hasOpenTrade)
        {
            G_SymbolTradeStates[symbol_idx].hasOpenTrade = false;
            G_SymbolTradeStates[symbol_idx].partialTp1Hit = false;
            G_SymbolTradeStates[symbol_idx].strategyInvolved = STRATEGY_NONE;
            G_SymbolTradeStates[symbol_idx].breakoutBarIndex = -1;
            G_SymbolTradeStates[symbol_idx].breakoutOccurredTime = 0;
        }
    }

    bool canTradePullback = InpUsePullbackTimeWindow? IsWithinTradingHours(InpPullbackStartTimeH, InpPullbackStartTimeM, InpPullbackEndTimeH, InpPullbackEndTimeM) : true;
    bool canTradeBreakout = IsWithinTradingHours(InpBreakoutStartTimeH, InpBreakoutStartTimeM, InpBreakoutEndTimeH, InpBreakoutEndTimeM);

    if(canTradePullback)
    {
        if(CheckPullbackEntry(symbol, symbol_idx))
            return; // Trade opened, move to next symbol
    }
    if(canTradeBreakout &&!G_SymbolTradeStates[symbol_idx].hasOpenTrade)
    {
        if(CheckBreakoutEntry(symbol, symbol_idx))
            return; // Trade opened
    }
}

bool CheckPullbackEntry(string symbol, int symbol_idx) // [18]
{

    ENUM_TREND_STATE trend = GetTrendState(symbol, InpTrendTimeframe, G_SymbolIndicatorHandles[symbol_idx].h_TrendEMA_TrendTF);
    if(trend == TREND_NONE_SIDEWAYS) return false;

    double vwap_value = GetVWAP(symbol, InpSignalTimeframe, symbol_idx);
    if(vwap_value == 0 || vwap_value == EMPTY_VALUE) return false;

    MqlRates rates_signalTF[];
    if(CopyRates(symbol, InpSignalTimeframe, 0, 3, rates_signalTF) < 3) return false; // Bar 0 (current), 1 (last closed), 2

    double current_ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double current_bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    MqlRates last_closed_candle_signalTF = rates_signalTF[1]; // Bar 1 is the last fully closed candle

    bool pullback_condition_met = false;
    double entry_price = 0;
    ENUM_ORDER_TYPE order_type = WRONG_VALUE;

    // VWAP Gap Rule from [18]: "If the price gaps more than 0.5% at the open, ignore the VWAP initially."
    // This needs to check daily open. There's potentially to add advanced gap rule but use simplicity for now
    if(trend == TREND_UP)
    {
        double vwap_pullback_level = vwap_value * (1.0 - InpVWAP_Proximity_Percent / 100.0);
        double last_hl_level = rates_signalTF[0].low; // Higher low of bar before last_closed_candle_signalTF

        
        if(last_closed_candle_signalTF.low <= vwap_pullback_level || last_closed_candle_signalTF.low <= last_hl_level)
        {
    
            if(current_ask > last_closed_candle_signalTF.high)
            {
                 // Trend invalid if price crosses VWAP opposite to trend [18]
                if(last_closed_candle_signalTF.close < vwap_value) return false;

                pullback_condition_met = true;
                entry_price = current_ask;
                order_type = ORDER_TYPE_BUY;
            }
        }
    }
    else if(trend == TREND_DOWN)
    {
        double vwap_bounce_level = vwap_value * (1.0 + InpVWAP_Proximity_Percent / 100.0);
        double last_lh_level = rates_signalTF[0].high; // Lower high of bar before last_closed_candle_signalTF
        
        if(last_closed_candle_signalTF.high >= vwap_bounce_level || last_closed_candle_signalTF.high >= last_lh_level)
        {
            // Price breaks below last M1 low after bounce
            if(current_bid < last_closed_candle_signalTF.low)
            {
                // Trend invalid if price crosses VWAP opposite to trend [18]
                if(last_closed_candle_signalTF.close > vwap_value) return false;

                pullback_condition_met = true;
                entry_price = current_bid;
                order_type = ORDER_TYPE_SELL;
            }
        }
    }

    if(!pullback_condition_met) return false;

    // --- Entry on Breakout with Volume (Signal Timeframe) ---
    // The "breakout" here refers to the breakout of the last M1 high/low after the pullback.
    // The volume check is for the candle that performs this small breakout (current forming bar, or last closed if entry is on next bar open)
    // [18]: "The breakout candle must have at least 1.4 times more volume..."
    // This should refer to `last_closed_candle_signalTF` if entry is on current_ask > last_closed_candle_signalTF.high
    
    long volumes_signalTF[];
    if(CopyTickVolume(symbol, InpSignalTimeframe, 0, InpVolumeMA_Period + 3, volumes_signalTF) < InpVolumeMA_Period + 3) return false;
    
    // Average volume of previous 15 candles (excluding the breakout candle itself)
    double avg_volume_prev_N = GetAverageVolume(symbol, InpSignalTimeframe, InpVolumeMA_Period, 2); // Start from index 2 (bar before last closed) for InpVolumeMA_Period bars
    if(avg_volume_prev_N <= 0) return false;

    // Volume of the "breakout candle" (last_closed_candle_signalTF which is volumes_signalTF)
    if(volumes_signalTF[1] < InpPullbackEntryVolumeFactor * avg_volume_prev_N) return false;

    // Volume increasing over last three candles (volumes_signalTF > volumes_signalTF > volumes_signalTF)
    if(!(volumes_signalTF[0] > volumes_signalTF[1] && volumes_signalTF[1] > volumes_signalTF[2])) return false;

    double stop_loss_price_val = CalculatePriceFromPercent(entry_price, InpPullbackStopLoss_Percent, (order_type == ORDER_TYPE_BUY? false : true));
    double take_profit_price_val = CalculatePriceFromPercent(entry_price, InpPullbackTakeProfit1_Percent, (order_type == ORDER_TYPE_BUY? true : false));

    double stop_loss_pips = MathAbs(entry_price - stop_loss_price_val) / SymbolInfoDouble(symbol, SYMBOL_POINT);
    double take_profit_pips = MathAbs(entry_price - take_profit_price_val) / SymbolInfoDouble(symbol, SYMBOL_POINT);

    if(stop_loss_pips < SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL) * 1.1) // Ensure SL is not too close, add 10% buffer
       stop_loss_pips = SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL) * 1.1;


    double lot_size = CalculateLotSize(symbol, stop_loss_pips);
    if(lot_size <= 0) return false;

    ulong ticket_nr = 0;
    string comment = "Pullback";
    if(OpenPosition(symbol, order_type, lot_size, entry_price, stop_loss_pips, take_profit_pips, comment, ticket_nr, symbol_idx))
    {
        G_SymbolTradeStates[symbol_idx].strategyInvolved = STRATEGY_PULLBACK;
        // SL/TP prices are stored by OpenPosition using actual fill price
        G_SymbolTradeStates[symbol_idx].ticket = ticket_nr;
        G_SymbolTradeStates[symbol_idx].tradeType = order_type;
        G_SymbolTradeStates[symbol_idx].openPrice = entry_price;
        G_SymbolTradeStates[symbol_idx].openTime = TimeCurrent();
        G_SymbolTradeStates[symbol_idx].initialStopLossPrice = stop_loss_price_val;
        G_SymbolTradeStates[symbol_idx].initialTakeProfitPrice = take_profit_price_val;
        G_SymbolTradeStates[symbol_idx].lotSize = lot_size;
        return true; // Trade opened
    }
    return false; // Trade not opened
}

bool CheckBreakoutEntry(string symbol, int symbol_idx) // [18]
{
    // --- Trend Identification (M5 chart - InpTrendTimeframe) ---
    ENUM_TREND_STATE trend = GetTrendState(symbol, InpTrendTimeframe, G_SymbolIndicatorHandles[symbol_idx].h_TrendEMA_TrendTF);
    if(trend == TREND_NONE_SIDEWAYS) return false;

   
    double breakout_level = GetBreakoutLevel(symbol, InpSignalTimeframe, G_SymbolIndicatorHandles[symbol_idx].h_TrendEMA_TrendTF);
    if(breakout_level == 0 || breakout_level == EMPTY_VALUE) return false;

    MqlRates rates_signalTF[];
    if(CopyRates(symbol, InpSignalTimeframe, 0, 3, rates_signalTF) < 3) return false; // Bar 0 (current), 1 (last closed), 2

    double current_ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double current_bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    MqlRates last_closed_candle_signalTF = rates_signalTF[1]; // Bar 1 is the last fully closed candle

    bool breakout_condition_met = false;
    double entry_price = 0;
    ENUM_ORDER_TYPE order_type = WRONG_VALUE;

    if(trend == TREND_UP)
    {
        if(current_ask > breakout_level)
        {
            breakout_condition_met = true;
            entry_price = current_ask;
            order_type = ORDER_TYPE_BUY;
        }
    }
    else if(trend == TREND_DOWN)
    {
        if(current_bid < breakout_level)
        {
            breakout_condition_met = true;
            entry_price = current_bid;
            order_type = ORDER_TYPE_SELL;
        }
    }

    if(!breakout_condition_met) return false;

    // --- Entry on Breakout with Volume (Signal Timeframe) ---
    // The "breakout" here refers to the breakout of the last M1 high/low after the pullback.
    // The volume check is for the candle that performs this small breakout (current forming bar, or last closed if entry is on next bar open)
    // [18]: "The breakout candle must have at least 1.4 times more volume..."
    // This should refer to `last_closed_candle_signalTF` if entry is on current_ask > last_closed_candle_signalTF.high
    
    long volumes_signalTF[];
    if(CopyTickVolume(symbol, InpSignalTimeframe, 0, InpVolumeMA_Period + 3, volumes_signalTF) < InpVolumeMA_Period + 3) return false;
    
    double avg_volume_prev_N = GetAverageVolume(symbol, InpSignalTimeframe, InpVolumeMA_Period, 2); // Start from index 2 (bar before last closed) for InpVolumeMA_Period bars
    if(avg_volume_prev_N <= 0) return false;

    // Volume of the "breakout candle" (last_closed_candle_signalTF which is volumes_signalTF)
    if(volumes_signalTF[1] < InpBreakoutEntryVolumeFactor * avg_volume_prev_N) return false;

    if(!(volumes_signalTF[0] > volumes_signalTF[1] && volumes_signalTF[1] > volumes_signalTF[2])) return false;

    double stop_loss_price_val = CalculatePriceFromPercent(entry_price, InpBreakoutStopLoss_Percent, (order_type == ORDER_TYPE_BUY? false : true));
    double take_profit_price_val = CalculatePriceFromPercent(entry_price, InpBreakoutTakeProfit_Percent, (order_type == ORDER_TYPE_BUY? true : false));

    double stop_loss_pips = MathAbs(entry_price - stop_loss_price_val) / SymbolInfoDouble(symbol, SYMBOL_POINT);
    double take_profit_pips = MathAbs(entry_price - take_profit_price_val) / SymbolInfoDouble(symbol, SYMBOL_POINT);

    if(stop_loss_pips < SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL) * 1.1) // Ensure SL is not too close, add 10% buffer
       stop_loss_pips = SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL) * 1.1;

    double lot_size = CalculateLotSize(symbol, stop_loss_pips);
    if(lot_size <= 0) return false;

    ulong ticket_nr = 0;
    string comment = "Breakout";
    if(OpenPosition(symbol, order_type, lot_size, entry_price, stop_loss_pips, take_profit_pips, comment, ticket_nr, symbol_idx))
    {
        G_SymbolTradeStates[symbol_idx].strategyInvolved = STRATEGY_BREAKOUT;
        // SL/TP prices are stored by OpenPosition using actual fill price
        G_SymbolTradeStates[symbol_idx].ticket = ticket_nr;
        G_SymbolTradeStates[symbol_idx].tradeType = order_type;
        G_SymbolTradeStates[symbol_idx].openPrice = entry_price;
        G_SymbolTradeStates[symbol_idx].openTime = TimeCurrent();
        G_SymbolTradeStates[symbol_idx].initialStopLossPrice = stop_loss_price_val;
        G_SymbolTradeStates[symbol_idx].initialTakeProfitPrice = take_profit_price_val;
        G_SymbolTradeStates[symbol_idx].lotSize = lot_size;
        return true; // Trade opened
    }
    return false; // Trade not opened
}

double GetVWAP(string symbol, ENUM_TIMEFRAMES timeframe, int handle_index)
{
    // Calculate daily VWAP using bar closes and volumes for the current day
    MqlRates rates[];
    int bars = CopyRates(symbol, PERIOD_M1, 0, 1440, rates); // Up to 1440 bars (1 day)
    if(bars <= 0)
        return 0.0;

    MqlDateTime dt;
    TimeToStruct(TimeCurrent(), dt);
    int today = dt.year * 10000 + dt.mon * 100 + dt.day;
    
    double sum_price_volume = 0.0;
    double sum_volume = 0.0;
    for(int i = 0; i < bars; i++)
    {
        MqlDateTime bar_dt;
        TimeToStruct(rates[i].time, bar_dt);
        int bar_date = bar_dt.year * 10000 + bar_dt.mon * 100 + bar_dt.day;
        if(bar_date == today)
        {
            sum_price_volume += rates[i].close * (double)rates[i].tick_volume;
            sum_volume += (double)rates[i].tick_volume;
        }
    }
    if(sum_volume == 0.0)
        return 0.0;
    return sum_price_volume / sum_volume;
}

double GetBreakoutLevel(string symbol, ENUM_TIMEFRAMES timeframe, int direction)
{
    // direction: 1 = buy (return previous bar high), -1 = sell (return previous bar low)
    MqlRates rates[2];
    if(CopyRates(symbol, timeframe, 0, 2, rates) < 2)
        return 0.0;
    if(direction == 1)
        return rates[1].high; // previous bar high
    else if(direction == -1)
        return rates[1].low;  // previous bar low
    return 0.0;
}

double GetAverageVolume(string symbol, ENUM_TIMEFRAMES timeframe, int period, int shift)
{
    // Calculate average tick volume over [period] bars, starting from [shift]
    if(period <= 0) return 0.0;
    MqlRates rates[];
    int bars_needed = period + shift;
    if(CopyRates(symbol, timeframe, 0, bars_needed, rates) < bars_needed)
        return 0.0;
    double sum = 0.0;
    for(int i = shift; i < shift + period; i++)
        sum += (double)rates[i].tick_volume;
    return sum / period;
}

double CalculatePriceFromPercent(double price, double percent, bool isBuy)
{
    double factor = 1.0 + (percent / 100.0);
    if(isBuy)
        return price * factor; // For TP on buy, SL on sell
    else
        return price / factor; // For SL on buy, TP on sell
}

double CalculateLotSize(string symbol, double stop_loss_pips)
{

    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double risk_per_trade = equity * InpRiskPercentPerTrade / 100.0;
    double tick_value = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
    double contract_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE);
    double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
    double sl_price_distance = stop_loss_pips * point;
    if(sl_price_distance <= 0.0 || tick_value <= 0.0 || contract_size <= 0.0)
        return 0.0;
    // Risk per lot = stop loss (in price) * contract size per lot
    double lot_size = risk_per_trade / (sl_price_distance * contract_size / point);
    double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
    double lot_step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
    lot_size = MathMax(min_lot, MathMin(max_lot, MathFloor(lot_size / lot_step) * lot_step));
    return lot_size;
}

bool OpenPosition(string symbol, ENUM_ORDER_TYPE order_type, double lot_size, double entry_price, double stop_loss_pips, double take_profit_pips, string comment, ulong& ticket, int handle_index)
{
    // Open a market order with given parameters using CTrade
    double sl = 0, tp = 0;
    if(order_type == ORDER_TYPE_BUY)
    {
        sl = entry_price - stop_loss_pips * SymbolInfoDouble(symbol, SYMBOL_POINT);
        tp = entry_price + take_profit_pips * SymbolInfoDouble(symbol, SYMBOL_POINT);
    }
    else if(order_type == ORDER_TYPE_SELL)
    {
        sl = entry_price + stop_loss_pips * SymbolInfoDouble(symbol, SYMBOL_POINT);
        tp = entry_price - take_profit_pips * SymbolInfoDouble(symbol, SYMBOL_POINT);
    }
    trade.SetExpertMagicNumber(InpMagicNumber);
    bool result = false;
    if(order_type == ORDER_TYPE_BUY)
        result = trade.Buy(lot_size, symbol, entry_price, sl, tp, comment);
    else if(order_type == ORDER_TYPE_SELL)
        result = trade.Sell(lot_size, symbol, entry_price, sl, tp, comment);
    if(result)
        ticket = trade.ResultOrder();
    return result;
}

ENUM_TREND_STATE GetTrendState(string symbol, ENUM_TIMEFRAMES timeframe, int ema_handle)
{
    double ema_50[2], ema_200[2];
    int bars = 2;
    int h_ema_50 = iMA(symbol, timeframe, 50, 0, MODE_EMA, PRICE_CLOSE);
    int h_ema_200 = iMA(symbol, timeframe, 200, 0, MODE_EMA, PRICE_CLOSE);
    if(h_ema_50 == INVALID_HANDLE || h_ema_200 == INVALID_HANDLE)
        return TREND_NONE_SIDEWAYS;
    if(CopyBuffer(h_ema_50, 0, 0, bars, ema_50) < bars || CopyBuffer(h_ema_200, 0, 0, bars, ema_200) < bars)
        return TREND_NONE_SIDEWAYS;
    int h_adx = iADX(symbol, timeframe, 10);
    double adx[2];
    if(h_adx == INVALID_HANDLE || CopyBuffer(h_adx, 0, 0, bars, adx) < bars)
        return TREND_NONE_SIDEWAYS;
    if(adx[1] < 20.0)
        return TREND_NONE_SIDEWAYS;
    if(ema_50[1] > ema_200[1])
        return TREND_UP;
    else if(ema_50[1] < ema_200[1])
        return TREND_DOWN;
    return TREND_NONE_SIDEWAYS;
}

bool IsWithinTradingHours(int start_hour, int start_minute, int end_hour, int end_minute)
{
    datetime now = TimeCurrent();
    MqlDateTime dt;
    TimeToStruct(now, dt);
    int current_minutes = dt.hour * 60 + dt.min;
    int start_minutes = start_hour * 60 + start_minute;
    int end_minutes = end_hour * 60 + end_minute;
    return (current_minutes >= start_minutes && current_minutes <= end_minutes);
}

void ManageOpenPosition(string symbol, TradeState& trade_state, int handle_index)
{
    double atr[1];
    int atr_handle = G_SymbolIndicatorHandles[handle_index].h_ATR_SignalTF;
    if(atr_handle != INVALID_HANDLE && CopyBuffer(atr_handle, 0, 0, 1, atr) == 1)
    {
        double trail_distance = atr[0];
        double price = (trade_state.tradeType == ORDER_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_BID) : SymbolInfoDouble(symbol, SYMBOL_ASK);
        double new_sl = 0;
        if(trade_state.tradeType == ORDER_TYPE_BUY)
            new_sl = price - trail_distance;
        else
            new_sl = price + trail_distance;
    
        if((trade_state.tradeType == ORDER_TYPE_BUY && new_sl > trade_state.initialStopLossPrice) ||
           (trade_state.tradeType == ORDER_TYPE_SELL && new_sl < trade_state.initialStopLossPrice))
        {
            double tp = trade_state.initialTakeProfitPrice; // Keep the original TP
            trade.PositionModify(symbol, new_sl, tp);
            trade_state.initialStopLossPrice = new_sl;
        }
    }
    if(TimeCurrent() - trade_state.openTime >= 3 * 60 * 60)
    {
        trade.PositionClose(trade_state.ticket);
        trade_state.hasOpenTrade = false;
        return;
    }
    if(trade_state.strategyInvolved == STRATEGY_PULLBACK && !trade_state.partialTp1Hit)
    {
        double tp1 = trade_state.initialTakeProfitPrice;
        double price = (trade_state.tradeType == ORDER_TYPE_BUY) ? SymbolInfoDouble(symbol, SYMBOL_BID) : SymbolInfoDouble(symbol, SYMBOL_ASK);
        if((trade_state.tradeType == ORDER_TYPE_BUY && price >= tp1) ||
           (trade_state.tradeType == ORDER_TYPE_SELL && price <= tp1))
        {
            double volume = trade_state.lotSize * 0.5;
            trade.PositionClosePartial(trade_state.ticket, volume);
            trade_state.partialTp1Hit = true;
        }
    }
}