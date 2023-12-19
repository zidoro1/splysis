# 概要
splatoon3のバトルメモリーからデータを採取し、統計解析を行う。

## 採取データ
自動化ツールで実行されるバトルメモリーの再生途中に、カウントグラフ、メインギアなどを画面表示させる。キャプチャーボードから得られた映像にテンプレートマッチングを行う。得られるデータの見込みは以下の通り。
- [x] リザルト画面。（年月日日時、マッチ区分、ステージ、ルール、ブキ種、塗りポイント、キル（＋アシスト）回数、デス回数、スペシャル使用回数）
- [ ] グラフ画像
- [ ] メインギア構成
- [x] プレイヤーの位置座標の時系列
- [ ] プレイヤーのスペシャル蓄積量の時系列

（注意）ロビー端末上にて、バトル戦績ボタンを選択する。メモリープレイヤーボタンからのリザルト画面は未対応。バトルメモリー自体はどちらでも良い。

### リザルト画面のみのデータ採取
プレイヤーの位置情報などを大量のフレームで処理するためGPUで並列化を行っている。
しかし、リザルト画面のみのデータ採取を目的とする場合はCPU版の関数を利用するほうが手間が少ないと思う。( get_match_cpu.py getResultCpu() )


## データ解析


## 自動化のコード
NX Macro Controll
"""
    後々アップロードする
"""
