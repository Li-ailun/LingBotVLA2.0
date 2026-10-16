# Full30K评测环境

本目录保存可移植环境定义和已验证DRLServer环境的文本快照，不保存Conda二进制包、模型或仿真资产。

权威环境只能在已经成功运行full30k RTC评测的DRLServer上导出：

```bash
cd /home/hddData/User/lixiang/LingBotVLA2.0_new
bash scripts/export_verified_environments.sh
```

导出器使用临时目录完成全部采集和净化后才替换目标文件，并拒绝将`/home/lixiang`、`/home/hddData`、`file://`或本地editable依赖写入可移植YAML。首次完成服务器导出前，不应根据本地工作站环境手工猜测依赖版本。

导出并提交两个YAML后，新服务器创建环境：

```bash
bash scripts/create_eval_environments.sh \
  --conda /path/to/conda \
  --lingbot-prefix /path/to/conda-envs/lingbotvla \
  --robotwin-prefix /path/to/conda-envs/RoboTwin
```

已有同名环境时默认拒绝覆盖；只有明确传入`--update`才会更新。先检查命令而不创建环境可追加`--dry-run`。
