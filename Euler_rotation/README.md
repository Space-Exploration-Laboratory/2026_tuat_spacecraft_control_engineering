# Euler rotation demonstration
## 概要

このPythonコードは、オイラー角を用いて姿勢変換を表記した場合の、オイラー角の選び方の違いによる結果の違いを可視化するものである。

## 使い方

```
python3 euler_rotation_animation.py --sequence ZYX --initial 0 0 0 --final 45 30 60
```
初期姿勢も指定するなら、
```
python3 euler_rotation_animation.py --sequence ZYX --initial 0 0 0 --final 45 30 60 --save attitude.gif
```
`--save`オプションで、結果をgifに出力できる。

オプションの説明は以下の通り。
```
--sequence    オイラー角列。例: ZYX, ZXZ, XYZ
--initial     初期オイラー角[deg]を3つ
--final       終端オイラー角[deg]を3つ
```

## 告知事項
このコードの生成には、Open AI の CodexおよびChat GPTが使用されている。