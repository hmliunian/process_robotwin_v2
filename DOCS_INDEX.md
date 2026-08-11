# 文档索引

- `docs/process_data_v2_architecture_design.md`：v2 三阶段架构设计（State Loop / Qwen
  Semantic Plan / SAM target-receiver）。
- `docs/process_data_v3_overview.md`：v3 变更概览，给人看，先读这个。
- `docs/process_data_v3_architecture_design.md`：v3 详细实现设计（gripper 并入 pipeline、
  脚本清理、一键处理任意目录），给实现者/AI 看。
- `docs/process_data_v3_1_architecture_design.md`：v3.1 增量设计（URDF gripper derived-run
  backend、统一 public artifact、resume 与依赖边界），加入/修改 URDF 模式时以此为准。
- `docs/process_data_v3_progress.md`：v3 各阶段（P1–P6）完成状态，继续实施前先看这个。
- `docs/gripper_pose_roi_coverage20_experiment.md`、`docs/qwen_mask_gripper_fixed_bbox_
  experiment.md`、`docs/qwen_mask_gripper.md`、`docs/video_mask_tracking_experiment.md`：
  gripper ROI 和跟踪方案的选型实验记录，是 v3 设计中参数取值的依据，非当前架构本身。
- `docs/urdf_gripper_mask_coverage20_experiment.md`：URDF gripper 的 coverage20 实验、命令、
  已验证产物与运行记录，是 v3.1 的实证配套文档。
- `README.md`：范围、目录和常用入口。
- `QUICKSTART.md`：最短运行步骤。

旧的 keyframe review、人工审批、独立 gripper 脚本和 box 架构已经从工作树删除，仍可从
Git 历史查看。
