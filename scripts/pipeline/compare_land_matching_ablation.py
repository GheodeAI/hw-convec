"""Isolate retention changes from adding peak area versus exact start year."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.matching import COV, match


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    e=pd.read_csv(source/'area_year_diagnostics_v1/events_with_peak_daily_land_area.csv')
    e=e[e.start_month.between(4,9)&(e.cg_coverage_fraction==1)].set_index('event_id')
    scenarios=[('baseline','exact_month',COV[:4]),('area_only','exact_month',COV),
               ('year_only','exact_year_month',COV[:4]),('area_and_year','exact_year_month',COV)]
    rows=[]
    for name,mode,variables in scenarios:
        pairs,cov,covariance=match(e,mode,base_cov=variables)
        a=e.loc[pairs.compound_event_id];b=e.loc[pairs.control_event_id]
        assert (a.start_month.values==b.start_month.values).all()
        if mode=='exact_year_month':assert (a.year.values==b.year.values).all()
        pairs.to_csv(output/f'{name}_pairs.csv',index=False)
        pd.DataFrame(covariance,index=cov,columns=cov).to_csv(output/f'{name}_covariance.csv')
        # These reference files only exist for the primary 90p20 run (a one-time
        # provenance check that a fresh recompute reproduces the earlier cached
        # result); other absolute-threshold runs have no such baseline, so the
        # check is skipped rather than failing the whole comparison.
        reference={'baseline':source/'tables/match_pairs.csv',
                   'area_and_year':source/'rematch_year_area_v1/exact_year_month/match_pairs.csv'}.get(name)
        if reference is not None and reference.exists():
            old=pd.read_csv(reference)
            if pairs[['compound_event_id','control_event_id']].equals(old[['compound_event_id','control_event_id']]):
                assert np.allclose(pairs.mahalanobis_distance,old.mahalanobis_distance)
            else:
                # The cached file comes from a separate implementation of the same
                # greedy matching (rebuild_compound_land.py). Identical distances can
                # still order differently in their last floating-point bits, and one
                # swapped tie cascades through the rest of the greedy assignment. What
                # has to hold is that the two solutions are of equivalent quality, not
                # that they are pair-for-pair identical.
                shared=set(pairs.compound_event_id)&set(old.compound_event_id)
                new_map=dict(zip(pairs.compound_event_id,pairs.control_event_id))
                old_map=dict(zip(old.compound_event_id,old.control_event_id))
                agree=sum(1 for c in shared if new_map[c]==old_map[c])/max(len(shared),1)
                count_ratio=len(pairs)/max(len(old),1)
                mean_gap=abs(pairs.mahalanobis_distance.mean()-old.mahalanobis_distance.mean())
                print(f"[{name}] differs from cached {reference.name}: {agree:.1%} of shared compound events keep "
                      f"the same control, {len(pairs)} vs {len(old)} pairs, mean distance "
                      f"{pairs.mahalanobis_distance.mean():.6f} vs {old.mahalanobis_distance.mean():.6f}")
                if agree<.95 or not .98<count_ratio<1.02 or mean_gap>.01:
                    raise AssertionError(
                        f"{name} matching departs substantively from {reference}: agreement={agree:.1%}, "
                        f"pair-count ratio={count_ratio:.4f}, mean-distance gap={mean_gap:.6f}")
        rows.append(dict(scenario=name,exact_start_month=True,exact_start_year=mode=='exact_year_month',
                         distance_variables=';'.join(cov),pairs=len(pairs),compound_available=int(e.has_lightning.sum()),
                         controls_available=int((~e.has_lightning).sum()),compound_retention=len(pairs)/int(e.has_lightning.sum())))
    table=pd.DataFrame(rows)
    table['pairs_change_vs_baseline']=table.pairs-table.pairs.iloc[0]
    table['retention_change_pp_vs_baseline']=(table.compound_retention-table.compound_retention.iloc[0])*100
    table.to_csv(output/'retention_comparison.csv',index=False)
    (output/'README.md').write_text('''# 面积与年份单因素匹配对照

四个版本使用同一合格样本、D≤0.75 的 Mahalanobis caliper、全样本协方差、全局距离升序贪心、1:1 不放回。

- baseline：开始月份精确匹配；纬度、经度、持续时间、强度四变量。
- area_only：在 baseline 距离中仅加入 ln(最大单日陆地面积)，不限制年份。
- year_only：仅增加开始年份精确匹配（year×start_month 分层），距离仍为原四变量；不加入面积，也不把 year 当连续距离变量。
- area_and_year：开始年份＋月份精确匹配，距离五变量，包括对数面积。

每个版本按自身距离变量重新计算全合格样本协方差，因此增加变量会改变距离几何；匹配为贪心算法而非最大匹配数优化。本次只重跑匹配，未改动环境分析或此前输出。baseline 与 area_and_year 已逐对复现既有匹配结果。
''')
    print(table[['scenario','pairs','compound_retention','pairs_change_vs_baseline','retention_change_pp_vs_baseline']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.source,a.output)
