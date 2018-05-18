import pandas as pd
import alphalens as al
import matplotlib.pyplot as plt

def price_reader(price_path):
    price_df = pd.read_csv(price_path)
    price_df.rename(index=str,columns={"Unnamed: 0":"date"},inplace=True)
    price_df.date = pd.to_datetime(price_df.date,format="%Y-%m-%d",errors='ignore')
    # price_df.date = price_df.date.apply(timezone.localize)
    price_df.set_index(['date'],drop=True,inplace=True)
    price_df = price_df.sortlevel(axis=1)
    return price_df

def instrument_reader(instrument_path):
    instrument_df = pd.read_csv(instrument_path)
    instrument_df.drop(['Unnamed: 0'],axis=1,inplace=True)
    instrument_df = instrument_df.set_index(['bookId'])
    instrument_df = instrument_df.sort_index()
    return instrument_df

def equity_reader(equity_path):
    cn_df = pd.read_csv(equity_path)
    cn_df.date = pd.to_datetime(cn_df.date,format="%Y-%m-%d",errors='ignore')
    cn_df.set_index(['date','order_book_id'],drop=True,inplace=True)
    cn_df.drop(["Unnamed: 0"],axis=1,inplace=True)
    return cn_df

def equity_add_instrumentInfo(cn_df,instrument_df,instrument_column):
    instrumentInfoSeries = instrument_df[instrument_column]
    bookIdIdx = cn_df.index.get_level_values('order_book_id')
    bookIdArray = bookIdIdx.get_values()
    instrumentInfo = instrumentInfoSeries[bookIdArray[:]].values
    cn_df[instrument_column] = instrumentInfo
    return cn_df

def get_price_instrument_equity(price_path,instrument_path,equity_path,addInstrumentColumn=None):
    price_df = price_reader(price_path)
    instrument_df = instrument_reader(instrument_path)
    equity_df = equity_reader(equity_path)
    if(addInstrumentColumn):
        equity_df = equity_add_instrumentInfo(equity_df,instrument_df,addInstrumentColumn)
    return price_df,instrument_df,equity_df

def ic_analysis(equity_df, price_df, factor_columns, group_column, periods=(1,22,66), group_adjust=False):
    factor_list = []
    ic_list = []
    monthly_ic_list = []
    groupby = equity_df[group_column]
    for col in factor_columns:
        factor_list.append(equity_df[col])

    for my_factor in factor_list:
        factor_data = al.utils.get_clean_factor_and_forward_returns(factor=my_factor,
                                                                    prices=price_df,
                                                                    groupby=groupby,
                                                                    periods=periods,
                                                                    max_loss=1)
        mean_ic = al.performance.mean_information_coefficient(factor_data, group_adjust=group_adjust,
                                                              by_group=True,
                                                              by_time=None)
        mean_monthly_ic = al.performance.mean_information_coefficient(factor_data, group_adjust=group_adjust,
                                                                      by_group=False,
                                                                      by_time='M')
        print("#######################################################")
        print("factor: {}".format(my_factor.name))
        print(mean_ic)
        # print(mean_monthly_ic)
        ic_list.append(mean_ic)
        monthly_ic_list.append(mean_monthly_ic)
        al.plotting.plot_monthly_ic_heatmap(mean_monthly_ic)
        plt.show()


    mean_ic_df = pd.concat(ic_list, keys=factor_columns)
    mean_ic_df.index = mean_ic_df.index.set_names(['factor', 'group'])
    return mean_ic_df, monthly_ic_list