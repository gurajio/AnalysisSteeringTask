今回の実行: 2026-11-14、数値計算1コア。既存UI・CSV・過去の結果は変更していない。
結果保存先: /Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_paired_20261114/rerun_20261114_144205

ファイル
run_analysis.py: 承認された条件内t検定、補助Wilcoxon、各方法の2条件Holm補正、診断図。
analysis_snapshot.py: 実行時の既存CSVローダーのスナップショット。
verify_analysis.py: 元CSVの別読み取りによる採用判定・平均値・CI・Holm・入力ハッシュ照合。
protocol.json: 実行した計画（過去結果閲覧後の再解析）。
report_ja.txt: 結果、解釈の限界、除外集計。
participant_means.csv: 参加者別前後平均・有効件数。
trial_audit.csv: 対象区間全記録の出典行と採用・除外。
exclusion_summary.csv: 条件・区間別除外理由集計。理由の重複を別記。
coverage.csv: 元の試行番号別の有効MTの欠落と重複。
manifest.json / verification.json: 環境・入力ハッシュ・照合結果。
requirements.txt: 現在のPython 3.13環境の依存関係。

再実行（既存結果への上書きは拒否されるため、新しい空ディレクトリへコードのみコピーする）:
cd '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask'
run_dir=$(mktemp -d '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_paired_20261114/rerun_XXXXXXXX')
cp '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_paired_20261114/rerun_20261114_144205/run_analysis.py' '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_paired_20261114/rerun_20261114_144205/analysis_snapshot.py' '/Users/kazuho/Desktop/Programming/AnalysisSteeringTask/output/c2_c3_mt_paired_20261114/rerun_20261114_144205/verify_analysis.py' "$run_dir/"
./.venv/bin/python -B "$run_dir/run_analysis.py"
./.venv/bin/python -B "$run_dir/verify_analysis.py"

コードは親ディレクトリのinputCSVを検出する。別場所で実行する場合はSTEERING_ROOTでプロジェクトルートを指定できる。
run_analysis.pyは既存のoutput/c2_c3_validation_20261113/manifest.jsonとparticipant_summary.csvを追加照合に使う。
数値結果はresults.jsonとtests.csv。再実行では図の日時とmanifestの作成日時が変わる。
report_ja.txt末尾の今回の除外集計はexclusion_summary.csvから確認可能。
