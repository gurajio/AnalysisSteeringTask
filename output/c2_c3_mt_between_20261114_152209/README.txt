CoreMTのC2とC3の条件間比較

結果保存先: /Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_between_20261114_152209
主分析: 参加者別の後期301〜400平均−初期1〜100平均の条件間差。
副次分析: 参加者別の後期301〜400平均の条件間差。
差はC2−C3。両側Welchのt検定2件にHolm補正。補助検定なし。

ファイル
report_ja.txt: 計画・採用条件・主副の検定結果・除外集計・解釈・診断所見をまとめたレポート。
protocol.json: 実行前に保存した今回の計画。既存結果閲覧後の再解析であり、収集前事前登録ではない。
run_analysis.py / analysis_snapshot.py: 解析コードと実行時の既存CSVローダーのスナップショット。
participant_means.csv: 各参加者の前後平均、変化量、有効件数、元の番号の欠落。
trial_audit.csv: 対象区間の全記録の出典・採用・除外判定。
exclusion_summary.csv: 条件・区間別の除外理由と採用件数。理由の重複を別計上。
descriptives.csv: 各条件の参加者平均値と参加者間SD。
tests.csv / results.json: 2検定の数値結果。CIは個別・未補正。
PNG2枚: 変化量・後期平均の参加者全点と条件別Q–Q図。1600×1000px。
manifest.json: 入力ハッシュ、環境、検証結果。prior_checkは前回の条件内解析と参加者別平均が一致したことを示す。
requirements.txt: 既存Python環境の依存バージョン。新しい依存関係の追加なし。

再実行（同一結果を上書きしないため、新しい出力先を指定）
cd '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask'
new_out=$(mktemp -d '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_between_rerun_XXXXXXXX')
./.venv/bin/python -B '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_between_20261114_152209/run_analysis.py' --root '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask' --out "$new_out"

小さな既知値による検証のみ
./.venv/bin/python -B '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_between_20261114_152209/run_analysis.py' --check

数値計算は1コアに固定。入力CSVと既存UI・過去の解析結果は変更していない。
保存後、結果閲覧後の解釈上の注意と今回の計画をプロジェクトAGENTS.md末尾へ追記する。
参考となる過去の解析結果への照合は、対応ファイルが存在し入力ハッシュが一致する場合のみ行う。
今回の2検定への補正は、過去の分析選択全体への補正ではない。既存の広い検定群を変更しない。
