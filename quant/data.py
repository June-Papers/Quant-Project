"""Data loader utilities for market and fundamental data.

Assumes large parquet datasets are stored outside the project at ../data/.
Each dataset is expected to have a 'date' column and stock code columns.
"""
from pathlib import Path
import pandas as pd
from typing import Optional


class DataLoader:
    def __init__(self, data_path: Optional[str] = "../data"):
        self.data_path = Path(data_path)

    def _read(self, name: str) -> pd.DataFrame:
        p = self.data_path / name
        df = pd.read_parquet(p)
        # ensure date column
        if "date" not in df.columns:
            # if date is index
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={df.index.name: "date"})
            else:
                raise ValueError("Dataset must contain 'date' column or datetime index")
        return df

    def load_close(self) -> pd.DataFrame:
        return self._read("close.parquet")

    def load_cap(self) -> pd.DataFrame:
        return self._read("cap.parquet")

    def load_sector(self) -> pd.DataFrame:
        return self._read("sector.parquet")

    def load_volume(self) -> pd.DataFrame:
        return self._read("volume.parquet")

    def load_shares_outstanding(self) -> pd.DataFrame:
        return self._read("shares_outstanding.parquet")

    def load_halted(self) -> pd.DataFrame:
        return self._read("halted.parquet")

    def load_bs(self) -> pd.DataFrame:
        """Load Balance Sheet data (날짜는 컬럼명으로 존재, 'date' 컬럼 없음)."""
        p = self.data_path / "BS.parquet"
        return pd.read_parquet(p)

    def load_pl(self) -> pd.DataFrame:
        """Load P&L data (날짜는 컬럼명으로 존재, 'date' 컬럼 없음)."""
        p = self.data_path / "PL.parquet"
        return pd.read_parquet(p)

    def get_financial_data(
        self,
        df: pd.DataFrame,
        account_name: str,
        close_df: pd.DataFrame,
        apply_lag: bool = True,
        daily_fill: bool = True
    ) -> pd.DataFrame:
        """재무데이터 조회 및 처리 함수.
        
        Parameters
        ----------
        df : pd.DataFrame
            BS 또는 PL 데이터 (메타컬럼: 코드, 코드명, 아이템코드, 아이템명)
        account_name : str
            조회할 계정명 (예: '자산총계')
        close_df : pd.DataFrame
            일별 종가 데이터 ('date' 컬럼 포함)
        apply_lag : bool
            look-ahead bias 방지 lag 적용 여부
        daily_fill : bool
            거래일 기준 일별 확장(ffill) 여부
            
        Returns
        -------
        pd.DataFrame
            date 기준 일별 재무데이터
        """
        # 1. 계정 조회
        temp = df[df['아이템명'] == account_name].copy()
        if temp.empty:
            raise ValueError(f"'{account_name}' 계정을 찾을 수 없습니다.")

        meta_cols = ['코드', '코드명', '아이템코드', '아이템명']
        date_cols = [col for col in temp.columns if col not in meta_cols]

        temp = temp.melt(
            id_vars=['코드'],
            value_vars=date_cols,
            var_name='date',
            value_name='value'
        )
        temp['date'] = pd.to_datetime(temp['date'])

        result = (
            temp.pivot(
                index='date',
                columns='코드',
                values='value'
            )
            .sort_index()
            .reset_index()
        )

        # 2. Lag 적용
        if apply_lag:
            trading_days = (
                pd.to_datetime(close_df['date'])
                .sort_values()
                .unique()
            )

            def get_rebalancing_date(d):
                year = d.year
                month = d.month
                # 분기별 반영 시점
                if month == 3:
                    target = pd.Timestamp(year, 8, 31)
                elif month == 6:
                    target = pd.Timestamp(year, 11, 30)
                elif month == 9:
                    target = pd.Timestamp(year + 1, 2, 28)
                elif month == 12:
                    target = pd.Timestamp(year + 1, 5, 31)
                else:
                    return pd.NaT

                valid_days = trading_days[trading_days <= target]
                return valid_days[-1] if len(valid_days) > 0 else pd.NaT

            result['date'] = result['date'].apply(get_rebalancing_date)
            result = (
                result.groupby('date', as_index=False)
                .last()
                .sort_values('date')
            )

        # 3. 거래일 기준 일별 확장
        if daily_fill:
            trading_calendar = (
                pd.DataFrame({'date': pd.to_datetime(close_df['date'])})
                .drop_duplicates()
                .sort_values('date')
            )
            result = (
                trading_calendar
                .merge(result, on='date', how='left')
                .sort_values('date')
            )
            result = result.ffill()

        result = result.reset_index(drop=True)
        return result
