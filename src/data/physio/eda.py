import polars as pl
import neurokit2 as nk

from src.data.transform_data import map_trials

@map_trials
def nk_eda_process(df: pl.DataFrame, sampling_rate) -> pl.DataFrame:
    array_np = df.select('EDA_RAW').to_numpy().flatten()
    # NOTE: TODO: same output as nk.eda_phasic(nk.eda_process(array_np, sampling_rate=sampling_rate))
    # where eda_phasic only returns phasic and tonic components and nothing else
    # this might be more performant - for visualization purposes, we use nk.eda_process here
    # because it does not scale the tonic component to the same scale as the phasic component
    df_pd, info = nk.eda_process(array_np, sampling_rate=sampling_rate, method='neurokit')
    df_pd.rename(columns={
        'EDA_Tonic': 'nk_EDA_Tonic',
        'EDA_Phasic': 'nk_EDA_Phasic'
        }, inplace=True)
    df_to_add = pl.from_pandas(df_pd[['nk_EDA_Tonic', 'nk_EDA_Phasic']])
    df = df.hstack(df_to_add)
    return df

