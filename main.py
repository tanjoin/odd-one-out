# ==============================================================================
# Pipenv による環境構築方法
# ==============================================================================
# 1. 任意の作業ディレクトリで以下のコマンドを実行し、仮想環境を作成して必要なパッケージをインストールします。
#    $ pipenv install opencv-python numpy
#
# 2. 仮想環境のシェルに入ります。
#    $ pipenv shell
#
# 3. 本スクリプトを実行します（例: 入力画像「input.jpg」、マッチング用「template.png」）。
#    $ python main.py
# ==============================================================================

import cv2
import numpy as np


def invert_all_matched_areas(
    image_path, template_path, output_path="result.png", threshold=0.8
):
    """入力画像からテンプレート画像と一致するすべての領域（複数箇所）を探し、

    それらの領域をすべてネガポジ反転する関数

    Args:
        image_path (str): 入力画像のパス
        template_path (str): マッチング用（テンプレート）画像のパス
        output_path (str): 結果画像の保存先パス
        threshold (float): マッチングの閾値（0.0 〜 1.0）。値が大きいほど厳密にマッチングします
    """
    # 1. 画像の読み込み
    img = cv2.imread(image_path)
    template = cv2.imread(template_path)

    if img is None:
        print(f"エラー: 入力画像 {image_path} を読み込めませんでした。")
        return
    if template is None:
        print(
            f"エラー: マッチング用画像 {template_path} を読み込めませんでした。"
        )
        return

    # テンプレートのサイズを取得
    h, w = template.shape[:2]

    # 2. テンプレートマッチングの実行 (グレースケールに変換して処理)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # 規格化相互相関 (TM_CCOEFF_NORMED) を使用
    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # 閾値以上のマッチング候補位置をすべて取得
    loc = np.where(result >= threshold)
    scores = result[loc]

    # マッチング箇所の座標をリスト化 (x, y)
    candidates = list(zip(loc[1], loc[0]))

    if len(candidates) == 0:
        print(
            f"マッチングする箇所が見つかりませんでした。（閾値: {threshold}）"
        )
        return

    # 3. 重複検出の排除 (Non-Maximum Suppression: NMS)
    # 同じオブジェクトに対して少しズレた枠が複数検出されるのを防ぐため、
    # OpenCV標準のNMS処理を行い、最も確からしい位置だけを残します。
    boxes = [[int(x), int(y), int(w), int(h)] for (x, y) in candidates]
    confidences = [float(score) for score in scores]

    # nms_threshold（重なり許容度）: 0.5 以下にすることで重複した矩形を完全に排除します
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, score_threshold=threshold, nms_threshold=0.3
    )

    # 4. マッチングしたすべての領域をネガポジ反転
    result_img = img.copy()
    match_count = 0

    # NMSで生き残ったインデックスに対して処理
    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, box_w, box_h = boxes[idx]

            # 検出領域をネガポジ反転 (255から引く)
            result_img[y : y + box_h, x : x + box_w] = (
                255 - result_img[y : y + box_h, x : x + box_w]
            )
            match_count += 1
            print(
                f"マッチ位置 {match_count}: (x: {x}, y: {y}, スコア: {confidences[idx]:.4f}) の領域を反転しました。"
            )

    # 5. 結果の保存
    cv2.imwrite(output_path, result_img)
    print(f"合計 {match_count} 箇所の領域を反転しました。")
    print(f"結果画像を {output_path} に保存しました。")


if __name__ == "__main__":
    # 使用例: 引数に入力画像とマッチング用（テンプレート）画像のパスを指定して実行します
    input_image = "input.jpg"
    template_image = "template.png"

    # デフォルトの閾値は 0.8 ですが、検出漏れがある場合は下げ、誤検出がある場合は上げてください。
    match_threshold = 0.8

    print(
        f"「{input_image}」から「{template_image}」とマッチするすべての箇所を探して反転します..."
    )
    invert_all_matched_areas(
        input_image, template_image, "result.png", threshold=match_threshold
    )