# process_data_v2 v3 变更概览

v2 跑通了 target/receiver 的三阶段流程；gripper mask 的算法也做出来了，但一直是独立于
pipeline 之外的实验脚本。v3 要做四件事，把项目收敛回"一套流程、一个入口"。

## 1. gripper mask 接入正式 pipeline

现状：gripper 的 ROI 投影、候选生成、Qwen 选择这套算法在 `src/robotwin_annotation_v2/
experiments/` 里，跑法是单独的 `scripts/generate_gripper_mask_video_qwen_qc.py`，产出自己
的 `gripper_masks.npz`，渲染视频时再靠 `--gripper-mask-root` 事后拼进最终结果。

v3 之后：gripper 变成 `run_target_receiver.py` 的第四个子命令（`gripper`），跟在 target/
receiver 的 `sam` 阶段后面跑，结果直接写进同一份四通道 `masks.npz`，不再需要事后合并。

## 2. 清理没用的脚本和测试

三个只在选型阶段用过一次的 ROI 参数对比脚本（`analyze_gripper_roi_variants.py` 等）删掉，
结论已经在文档里；配套的单元测试一起删。`tests/contract/` 目录从建立起就是空的，也删。

## 3. review sheet 不再单独跑一步

`build_tracking_review_sheets.py` 现在要在渲染视频之后手动再跑一次。v3 把它并进
`render_coverage20_videos.py`，渲染完自动生成 review sheet，省一步。

## 4. 一键处理任意数据目录

现在所有命令都要求 episode 在写死的 20 条名单里。v3 新增 `scripts/process_dataset.py`，
指向任意 RoboTwin 格式目录（`data/chunk-*/episode_*.parquet` 结构），自动发现里面所有
episode，一条命令跑完 loop → qwen → sam → gripper → 渲染。`justfile` 加一条
`just process <dataset_root>`。

一键命令假定 Qwen server 已经在另一个终端跑着（`just serve-qwen`），不负责拉起它。

## 影响范围

- 不改动 target/receiver 已验证的核心逻辑（`_run_role`、Qwen semantic plan、temporal QC
  全部不动）；
- `masks.npz` 格式不变，只是 gripper 两个通道从"总是 not_annotated"变成"有结果就填、没有
  就仍是 not_annotated"，向后兼容；
- 需要在 GPU 上重跑 smoke episode 验证四通道产物，其余靠现有单元测试覆盖。

详细的文件级改动、新接口签名、提交顺序见
[`process_data_v3_architecture_design.md`](process_data_v3_architecture_design.md)。
