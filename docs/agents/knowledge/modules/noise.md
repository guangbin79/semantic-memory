# 任务事件噪声闸（noise）

> 凭据硬拒与派发页回显两族判定：安全族独立短路（不受 400 字符/状态关键词双条件限制）；判别精度与误杀面的实证校准记录。

## 凭据硬拒正则加固清单（2026-11-03 825ecc1）
- IGNORECASE 必须：`TOKEN:` 大写变体是日志/环境变量常态，敏感键小写敏感即 bypass（终闸 C 册实测）。
- 键值两侧引号包装：JSON 形态 `"token": "<40+位>"` 冒号前后各有引号，`token[:：]` 裸匹配失配放行——正则须 `[\"']?token[\"']?\s*[:：]\s*[\"']?`。
- pypi-AgEI 用子串锚不用串首锚：`.pypirc` 落盘形态 `password = pypi-AgEI…` 前导非空即绕过 `startswith`。
- token 真值不入 git 史：测试钉用形态合成串（首尾保留形态锚、中段合成）——真样本随 commit 入史不可撤回（终闸 C 册判例）。
- 边界残留（记录未修）：JWT `token: eyJ…` 首段遇 `.` 失配放行——不属已声明两族，扩字符类是范围决策。

## 回显族判别式：行级纯路径（2026-11-03 825ecc1）
- Chosen: `^page:\s*\S+\s*$`／`^env:\s*\S+\s*$` 行级匹配＋`[agora]` 头＋<500。
- Alternatives: 条级含 `"page:"`＋`"env:"` 子串（首版）——终闸 B 册实证误杀「正文同行提及 page:/env: 的 peer 知识附言」（部署笔记常态形态，触发串实测整条静默丢弃）。
- Reason: BOOT 回显的 page:/env: 恒为纯路径独立行，行级判别同拦截力＋零提及误杀。
- Tradeoff: 无空白连写提及行理论上可撞纯路径形态；独立 env 行双条件兜底，残留记录未修。

## 凭据行级硬滤与无裁判直存残留（2026-11-03 825ecc1）
- strip_noise_lines 已收编 `_is_credential` 行级硬滤——混合条目的蒸馏输入/释义侧凭据行被滤（挂点②③）。
- 残留（记录未修）：flush_distill_queue 无裁判直存路径（lifecycle.py entry=None 分支）存 `r["content"]` 原文，混合条目凭据行仍逐字落库；滤洗需改直存语义（判重键/原文一致性），超单任务范围留追单。

## 夜扫兜底清扫（2026-11-03 ag-bb3029，99d5e0e..7b868c4）
- 形态：lifecycle.noise_sweep(dry_run) 全库活条目过确定性谓词→forget 软删（purge 前可逆）；三弱信号齐（[agora] task 前缀＋裸40hex＋状态词）谓词未命中→只进晨报 suspicious 段。调度常驻 daemon 每日定点（缺省 03:30，MEMORY_NOISE_SWEEP_AT），单飞与 flush 同款；端点 POST /api/maintenance/noise_sweep 鉴权与 purge 同款双闸；黑名单规则升级不自动化（过拟合静默误杀走派发评审）。
- **安全族谓词单一真源（终闸 B/C 收敛 warning 判例）**：凡「按凭据族删除」的条目，其报告摘要省略判定必须复用 noise.py 的 `_is_credential`（全形态含 JSON 引号包装），不得用自有窄正则——两实现漂移曾致引号包装存量凭据原文经晨报落盘 logs/（记忆库外永不清除）。判据：删除谓词与脱敏谓词取并集时，以超集者为准。
- 软删期间分页扫防跳页：deleted=0 窗口前移量＝本页 forget 数，offset 回退等量补偿；他人并发删除已扫活行仍可滑过单行（天级延迟次日自愈，取舍注释在 lifecycle）。
- 晨报通道健壮性：daemon 先 log.info 后落盘报告文件（写盘失败报告仍留服务日志）；凭据命中条目摘要置 `[credential elided]` 占位。
