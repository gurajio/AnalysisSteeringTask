C2とC3のKruskal-Wallis追加探索分析

結果: report_ja.txt / results.json / tests.csv
対象値: participant_means.csv（前回のWelch比較と同じ20名。元CSVから照合済み）
順位: participant_ranks.csv
監査: trial_audit.csv / exclusion_summary.csv / manifest.json
計画: protocol.json
再現コード: run_analysis.py

後期301〜400試行の平均CoreMTと、後期平均−初期1〜100試行平均の変化量を参加者単位で比較。
両検定は同順位補正付きKruskal-Wallis H、自由度1のχ²近似p値。今回の2検定にHolm補正（5%）。
以前のWelch検定の結果を閲覧した後の追加探索分析として報告し、既存の分析を置き換えない。
今回の2検定のみの補正で、過去の分析選択全体の多重性を解消したとは扱わない。

再実行:
cd '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask'
new_out=$(mktemp -d '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_kruskal_rerun_XXXXXXXX')
./.venv/bin/python -B '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_kruskal_20261114_181133/run_analysis.py' --root '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask' --out "$new_out"

軽量検証のみ:
./.venv/bin/python -B '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_kruskal_20261114_181133/run_analysis.py' --check

既存のoutput/c2_c3_mt_between_20261114_152209内のparticipant_means.csv、trial_audit.csv、exclusion_summary.csv、manifest.json、checksums.jsonを照合元として使用する。
入力CSVハッシュが以前の解析時と異なる場合は停止し、古い集計値を黙って使用しない。
入力データ・既存結果・UIは変更しない。数値計算1コア。新しい依存関係なし。
