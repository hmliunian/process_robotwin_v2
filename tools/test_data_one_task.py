#!/usr/bin/env python3
"""快速测试脚本：从 data_one_task 读取数据并可视化。

这个脚本演示如何：
1. 读取 RoboTwin data_one_task 数据集
2. 解析 episode 信息
3. 提取视频帧
4. 可视化并保存
"""

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from robotwin_annotation_v2.domain import EpisodeRef


def load_episode_info(data_root: Path, episode_index: int):
    """加载 episode 元信息。"""
    episodes_file = data_root / "meta" / "episodes.jsonl"

    with open(episodes_file, "r") as f:
        for line in f:
            ep = json.loads(line)
            if ep["episode_index"] == episode_index:
                return ep

    raise ValueError(f"Episode {episode_index} not found")


def load_episode_parquet(data_root: Path, episode_index: int):
    """加载 episode parquet 文件。"""
    # data/chunk-000/episode_000000.parquet
    chunk = episode_index // 1000
    parquet_path = data_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    return df


def extract_frame_from_video(data_root: Path, episode_index: int, camera: str, frame_index: int):
    """从视频中提取指定帧。"""
    import av

    # videos/chunk-000/observation.images.cam_high/episode_000000.mp4
    chunk = episode_index // 1000
    video_key = f"observation.images.{camera}"
    video_path = data_root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{episode_index:06d}.mp4"

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 使用 av 解码
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]

    for i, frame in enumerate(container.decode(video_stream)):
        if i == frame_index:
            img = frame.to_image()
            container.close()
            return img

    container.close()
    raise ValueError(f"Frame {frame_index} not found in video")


def visualize_episode(data_root: Path, episode_index: int, output_dir: Path):
    """可视化一个 episode 的关键信息。"""
    print(f"\n{'='*60}")
    print(f"处理 Episode {episode_index:06d}")
    print(f"{'='*60}\n")

    # 1. 加载元信息
    ep_info = load_episode_info(data_root, episode_index)
    print(f"📋 Episode 信息:")
    print(f"  长度: {ep_info['length']} 帧")
    print(f"  任务数: {len(ep_info['tasks'])}")
    print(f"  第一个任务: {ep_info['tasks'][0]}")
    print()

    # 2. 加载 parquet
    df = load_episode_parquet(data_root, episode_index)
    print(f"📊 Parquet 数据:")
    print(f"  行数: {len(df)}")
    print(f"  列数: {len(df.columns)}")
    print(f"  列名: {', '.join(df.columns[:10])}...")
    print()

    # 3. 提取关键帧
    output_dir.mkdir(parents=True, exist_ok=True)

    # 提取开始、中间、结束帧
    key_frames = [0, ep_info['length'] // 2, ep_info['length'] - 1]

    for idx, frame_idx in enumerate(key_frames):
        print(f"📸 提取帧 {frame_idx}...")

        try:
            img = extract_frame_from_video(data_root, episode_index, "cam_high", frame_idx)

            # 添加标注
            draw = ImageDraw.Draw(img)
            text = f"Episode {episode_index:06d} | Frame {frame_idx:04d}"

            # 简单文字（PIL 默认字体）
            draw.text((10, 10), text, fill=(255, 255, 0))

            # 保存
            output_path = output_dir / f"episode_{episode_index:06d}_frame_{frame_idx:04d}.jpg"
            img.save(output_path, quality=95)
            print(f"  ✅ 保存到: {output_path}")

        except Exception as e:
            print(f"  ❌ 错误: {e}")

    print()


def main():
    """主函数。"""
    # 数据路径
    data_root = Path(__file__).parent.parent.parent / "process_data" / "data_one_task"
    output_dir = Path(__file__).parent.parent / "artifacts" / "data_one_task_viz"

    if not data_root.exists():
        print(f"❌ 数据目录不存在: {data_root}")
        sys.exit(1)

    print(f"📂 数据目录: {data_root}")
    print(f"📂 输出目录: {output_dir}")

    # 处理前 3 个 episodes
    for episode_idx in range(3):
        try:
            visualize_episode(data_root, episode_idx, output_dir)
        except Exception as e:
            print(f"❌ Episode {episode_idx:06d} 处理失败: {e}\n")

    print(f"\n{'='*60}")
    print("✅ 完成！")
    print(f"{'='*60}")
    print(f"\n查看输出: ls -lh {output_dir}/")


if __name__ == "__main__":
    main()
