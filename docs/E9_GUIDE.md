# E9：继续 / 停止 / 转向决策

E9 按附件第13节汇总 E4–E7 的真实证据，**不训练新模型、不修改网络或损失、不启动转向实验**。所有原数据路径和 E0–E8 实现保留。主判据基于 DIV2K ×4 的 E6 独立训练 seed 和 E7 逐图统计；E8 DF2K/多倍率结果可选，按训练数据和倍率单列，不能替换主零点。附件流程允许 E4–E7 不通过时直接进入 E9，因此不复用 E8 的“必须通过”入口锁。

## 判据与近似术语

精确条件保留原值：Case A 的五集等权平均 ΔPSNR ≥+0.05 dB、Urban100 >0、Manga109 >0；Case B 的 +0.02～+0.05 dB。先检查机制确已修复，再检查失败触发项、A、B，其他情况输出 `REVIEW`，不把所有情况强塞进三类。

附件中的近似术语使用下述**工程量化解释**，不是声称附件给出了这些额外数值。它们全部写入报告及 `options/E9_policy.example.json`，可在查看实测结果前固定自有策略；成功线0.05、弱成功区间0.02～0.05不可通过策略文件更改。

| 概念 | 默认实现 |
|---|---|
| uncertainty + local alpha 已修复 | 沿用 E4 选择器：U 的 Spearman >0.2，较 E0 提升≥0.05；A 的图内 std、P95−P05 均>0；E4/E6 的 U/A 一致，E5记录的U一致 |
| 3 seeds stable | 使用 E6 正式3个独立 seed（可选5个）；五集平均及 Urban100/Manga109 均为全部 seed 正增益；不能把一份权重的3次推理当3次训练 |
| Depth causality 成立／明显成立 | 正确Depth分别相对零Depth、打乱Depth的五集等权优势均 >0.01 dB，且所有 matched inference seed 差值均>0；这是保守量化 |
| avg≈+0.01 dB | 五集均值≤+0.015 dB，覆盖近零和负增益；0.015～0.02之间不自动归为失败 |
| Correct Depth≈Shuffle Depth | 五集差值绝对值≤0.01，且每个 matched inference seed 的绝对差均≤0.01；避免正负大幅抵消后误判“近似相等” |
| Urban100持续负 | 历史 E4 Urban100<0，E6所有独立seed<0，且E7图像bootstrap区间上界<0；若证据未充分则不认定持续负 |
| 个别dataset不稳定 | 任一数据集存在非正seed，或E7的95%图像区间下界≤0 |

数值比较采用1e-10 dB算术容差，避免浮点相减把精确0.05误分到下方。若机制已修复，低增益、Urban100持续负、正确≈打乱任一项成立则 Case C；另将“所有seed中打乱都不劣于正确”视为更强的因果反证，一并触发 C。A 要全部成功条件成立；B 要在弱成功区间、因果明显成立且存在不稳定数据集。恰好0.05先检查A，若未满足再检查B。高增益但不稳定、仅正向趋势、未修复机制或未落入条件等情况保留 `REVIEW`。

E7区间跨零只描述平均趋势，不称统计稳定改善。E6的跨seed稳定与E7的图像区间是不同证据；Case A也不自动等于全部数据集有显著图像级改善。报告逐集列出两类证据。E5是在单份原选定E4权重上做因果测试，不能由此声称所有E6/E8新权重都完成了因果复测。

旧E5报告记录了uncertainty模式，没有记录训练时的alpha正则标签；输入的E4选择文件必须确实属于该E5实验。脚本明确披露这项来源限制，不能声称仅凭旧E5报告就独立证明了alpha训练归属；可通过实文件hash检查核对所记录的E5权重。

## 输出和动作

- A：UDR-v2 finalization → DF2K → multi-scale → paper-scale benchmark → 再考虑Text。已有E8结果仅作为完成情况参考，不重复启动训练。
- B：优先研究geometry representation，对比D、∇D、∇²D、normal-like和multi-scale geometry，不继续增加fusion block。
- C：停止当前UDR复杂化，列出附件四条转向路线：Depth source、geometry representation、geometry-guided reconstruction、受空间约束的Text；不回到global CLIP FiLM。
- `REVIEW`：证据完整但不能完全匹配A/B/C，列出判据供复核。
- `insufficient_evidence`／`invalid_evidence`：缺失或不一致，不给研究方向成功／失败结论。

输出 `results/E9_decision/E9_decision.json`、`E9_summary.md`、`E9_criteria.csv`。JSON保存源文件路径与SHA256、策略、全部seed差值、CI、判据、动作及可选扩展证据。退出码0表示分析完成（包括C和REVIEW），**不表示研究成功**；退出码2表示证据缺失或无效。

## E9完整运行命令

使用原项目Python环境即可。数值复算依赖NumPy；无需GPU、PyTorch、BasicSR或Mamba扩展。

```bash
git clone https://github.com/KaiXu-HIT/v3.2-M3SR-MambaIRv2-E9.git
cd v3.2-M3SR-MambaIRv2-E9
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python scripts/udr/check_e9_decision.py
python scripts/udr/e9_decision.py --write-input-template options/E9_inputs.json
```

`options/E9_inputs.json` 内填真实报告和CSV路径。相对路径统一相对本项目根目录；也可以使用其他仓库的绝对路径。可以按下列方式设置（先按服务器实际位置调整root变量）：

```bash
E4ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E4
E5ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E5
E6ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E6
E7ROOT=/home/BRAIN/xukai/code/v3.2-M3SR-MambaIRv2-E7
export E4ROOT E5ROOT E6ROOT E7ROOT

python - <<'PY'
import json, os
from pathlib import Path
p = Path('options/E9_inputs.json')
config = json.loads(p.read_text(encoding='utf-8'))
config.update(
    selection=os.environ['E4ROOT']+'/experiments/E4_selection/E4_selection.json',
    e4_report=os.environ['E4ROOT']+'/results/E4_udrv2/ablation/E4_ablation.json',
    e5_report=os.environ['E5ROOT']+'/results/E5_depth_causality/E5_causality.json',
    e5_csv=os.environ['E5ROOT']+'/results/E5_depth_causality/E5_per_image.csv',
    e6_report=os.environ['E6ROOT']+'/results/E6_multiseed/E6_multiseed.json',
    e7_report=os.environ['E7ROOT']+'/results/E7_image_statistics/E7_statistics.json',
    e7_csv=os.environ['E7ROOT']+'/results/E7_image_statistics/E7_per_image.csv',
    e8_metrics=[])
p.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
PY

python scripts/udr/e9_decision.py \
  --inputs options/E9_inputs.json \
  --output results/E9_decision
```

脚本要求五数据集完整图像数5/14/100/100/109，校验CSV精确图像/seed/条件配对及有限数值，用逐图结果重算E6均值/样本std和E5对照差，用原E7 NumPy RNG流重算10,000次图像簇bootstrap并核对区间，检查E6/E7权重hash一致和跨训练seed权重互异。不只读取上游的“通过”布尔值。

只携带报告和CSV时，默认明确标记 `report_hash_linkage_only`，不声称已读取原权重文件。若原权重仍在报告记录路径，增加实文件SHA256检查：

```bash
python scripts/udr/e9_decision.py \
  --inputs options/E9_inputs.json \
  --verify-checkpoints \
  --output results/E9_decision_verified
```

若需事先固定不同的近似术语量化，复制并编辑策略；不要根据输出调门槛：

```bash
cp options/E9_policy.example.json options/E9_policy.json
python scripts/udr/e9_decision.py \
  --inputs options/E9_inputs.json \
  --policy options/E9_policy.json \
  --output results/E9_decision_policy
```

已有E8报告时，将其实际绝对路径逐项加入清单 `e8_metrics` 数组，例如×4/×2/×3各一份 `results/E8_scaling/xN/E8_metrics.json`，再执行同一命令。不要求先完成E8，不用DF2K对照替代DIV2K主决策，缺失E8因果复测也不冒充已完成。

## 需要重新生成测试证据时

E9没有训练命令。已有权重但需重新测试E6/E7时，可在E9项目中复用原脚本，原E6计划及其中记录的配置、选择文件和权重必须存在：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/udr/evaluate_e6_multiseed.py \
  --plan "$E6ROOT/experiments/E6_plan/E6_plan.json" \
  --output results/E9_recheck/E6

CUDA_VISIBLE_DEVICES=0 python scripts/udr/e7_image_statistics.py \
  --plan "$E6ROOT/experiments/E6_plan/E6_plan.json" \
  --e6-report results/E9_recheck/E6/E6_multiseed.json \
  --bootstrap-seed 2026 \
  --output results/E9_recheck/E7
```

将E9清单中的E6/E7报告及E7 CSV改为上述新输出路径后再运行决策。E4消融和E5因果报告仍须来自真实实验；如需重测，使用[E4指南](E4_GUIDE.md)、[E5指南](E5_GUIDE.md)中的完整命令。没有权重和实测证据时不自动启动新训练，先生成缺失项报告。
