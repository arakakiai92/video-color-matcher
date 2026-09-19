import subprocess
import tempfile
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="動画キャラクター色味統一ツール", page_icon="🎨", layout="centered"
)


def unify_character_color_stream(
    source_path, target_sample_path, final_output_path
):
  cap_src = cv2.VideoCapture(source_path)
  cap_tgt = cv2.VideoCapture(target_sample_path)

  ret_tgt, frame_tgt = cap_tgt.read()
  if not ret_tgt:
    return False

  fps = cap_src.get(cv2.CAP_PROP_FPS)
  width = int(cap_src.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap_src.get(cv2.CAP_PROP_FRAME_HEIGHT))
  total_frames = int(cap_src.get(cv2.CAP_PROP_FRAME_COUNT))

  # 一度OpenCVで書き出すためのテンポラリファイル
  raw_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

  progress_bar = st.progress(0)
  status_text = st.empty()
  frame_count = 0

  while cap_src.isOpened():
    ret_src, frame_src = cap_src.read()
    if not ret_src:
      break

    # 白背景を保護するためのマスク作成
    hsv_src = cv2.cvtColor(frame_src, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    bg_mask = cv2.inRange(hsv_src, lower_white, upper_white)
    char_mask = cv2.bitwise_not(bg_mask)

    # Lab色空間に変換
    src_lab = cv2.cvtColor(frame_src, cv2.COLOR_BGR2LAB).astype("float32")
    tgt_lab = cv2.cvtColor(frame_tgt, cv2.COLOR_BGR2LAB).astype("float32")

    src_pixels = src_lab[char_mask > 0]
    tgt_pixels = tgt_lab[char_mask > 0]

    if len(src_pixels) > 0 and len(tgt_pixels) > 0:
      src_mean = np.mean(src_pixels, axis=0)
      src_std = np.std(src_pixels, axis=0)
      tgt_mean = np.mean(tgt_pixels, axis=0)
      tgt_std = np.std(tgt_pixels, axis=0)

      adjusted_lab = src_lab.copy()
      for i in range(3):
        channel_data = adjusted_lab[:, :, i]
        std_src = src_std[i] if src_std[i] > 1e-5 else 1e-5

        masked_channel = (channel_data - src_mean[i]) * (
            tgt_std[i] / std_src
        ) + tgt_mean[i]
        channel_data = np.where(char_mask > 0, masked_channel, channel_data)
        adjusted_lab[:, :, i] = np.clip(channel_data, 0, 255)

      adjusted_lab = adjusted_lab.astype("uint8")
      adjusted_frame = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
    else:
      adjusted_frame = frame_src

    out.write(adjusted_frame)
    frame_count += 1
    if total_frames > 0:
      progress_bar.progress(min(frame_count / total_frames, 1.0))
      status_text.text(f"処理中... フレーム {frame_count} / {total_frames}")

  cap_src.release()
  cap_tgt.release()
  out.release()
  progress_bar.empty()
  status_text.empty()

  # ── ブラウザ再生可能なH.264形式にffmpegで変換 ──
  status_text.text("ブラウザ再生用に動画を最適化中...")
  try:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            raw_output_path,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            final_output_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    status_text.empty()
    return True
  except Exception as e:
    status_text.error(f"動画の変換に失敗しました: {e}")
    return False


# --- Streamlit UI ---
st.title("🎨 動画キャラクター色味統一ツール")
st.write(
    "基準となる動画の色味（本体・眉・尻尾などの茶色）を、変換したい動画のキャラクターに適用します。"
)

col1, col2 = st.columns(2)

with col1:
  st.subheader("1. 変換したい動画")
  source_file = st.file_uploader(
      "動画ファイルをアップロード", type=["mp4", "mov"], key="source"
  )

with col2:
  st.subheader("2. 色の基準にする動画")
  target_file = st.file_uploader(
      "動画ファイルをアップロード", type=["mp4", "mov"], key="target"
  )

if source_file and target_file:
  st.markdown("---")
  if st.button(
      "✨ 色味を統一する処理を開始", type="primary", use_container_width=True
  ):
    with st.spinner("動画のカラーマッチング処理を実行中..."):
      with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_src:
        tmp_src.write(source_file.read())
        src_path = tmp_src.name

      with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_tgt:
        tmp_tgt.write(target_file.read())
        tgt_path = tmp_tgt.name

      output_path = tempfile.NamedTemporaryFile(
          delete=False, suffix=".mp4"
      ).name

      success = unify_character_color_stream(src_path, tgt_path, output_path)

      if success:
        st.success("🎉 色味の統一処理が完了しました！")

        st.subheader("プレビュー（調整後）")
        st.video(output_path)

        with open(output_path, "rb") as f:
          st.download_button(
              label="📥 調整済み動画をダウンロード",
              data=f,
              file_name="unified_character_output.mp4",
              mime="video/mp4",
              use_container_width=True,
          )
