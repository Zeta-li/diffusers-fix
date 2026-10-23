# Value时间校正算法与执行协议

2026-10-22，优化阶段执行前冻结。**效果待测，不保证精确、有效或新颖。** 若主ROT仍有错字/分块等伪影，或净延迟下降未达10%，记录触发证据后执行本协议；不替换主阶段结果。

## 1 算法与假设

基于`value_time_correction.py`，仅校正目标图缓存的raw attention Value。旧R2、8步、CFG=1、首尾全图、BF16融合Lightning、SDPA及条件图/text每步重算均不变，不使用dense oracle或训练。

对每层，令n=LN(h)，u_t=1+scale(t)，b_t=shift(t)，则理想算术下：

`V_t = Wv(u_t⊙n_t+b_t)+bias`。

令a为该token最后真实计算的步，`p_t=to_v(b_t)`，其中含线性bias。假设旧归一化hidden近似可用，以已缓存真实V_a构造：

- SHIFT：`Vhat_t=V_a+p_t−p_a`。
- AFFINE：`α_ta=(u_t·u_a)/(||u_a||²+1e−12)`，`δ_ta=p_t−α_ta*p_a`，`Vhat_t=α_ta*V_a+δ_ta`。

α是每层/当前步/anchor步的一个标量，不是训练得到的通道矩阵；本轮不按结果裁剪α。强制对角α_tt=1、δ_tt=0，SHIFT同龄差值为0，刚刷新位置保持恒等。

校正只生成attention读取时的临时Value，转回原dtype；**不写回真实V anchor，不更新其真实时间戳，不把校正当刷新**。Key、attention权重、旧hidden和最终velocity不直接校正。当前真实选中token仍按原执行器重算。

冻结hidden及标量scale近似不成立时可失败；BF16运算也不具理想精确仿射性。`rho=||u_t−αu_a||/||u_t||`仅描述scale拟合，不能充当实际预测误差或质量保证。

## 2 静态表与冷成本

模型加载、融合并卸载adapter后，按固定8步模型时间一次性预计算temb、各层p、α、δ及rho。本基座60层×8步产生480次shift→to_v线性调用，**不是480次DiT**；temb仅8次，无图像教师查询。

表按FP32保存，记录prepare_seconds、table_bytes、各tensor形状、α/rho范围及权重/脚本hash。仅同权重、dtype、device和相同8步schedule可复用。稳态请求可排除冷表构建；首次请求另报“表准备+该请求”，模型加载再单列。额外表存储不计作显存节省。

## 3 矩阵与顺序

**F/R2/SHIFT/AFFINE ×3旧样例×（预热1+计时3）=48请求、384次真实DiT。** 含12预热、36计时，每请求8次，性能路径额外oracle为0。

F原生完整，R2无校正；SHIFT/AFFINE用相同旧R2预算与刷新时序，仅Value读取规则不同。使用原图、指令、seed、显式初始latent/hash，不改ROI或评价区。三个已查看示例属于后验机制研究，不是独立Test。

使用独立`run_optimization.py`；主runner不修改。先全部case/config预热，随后按repeat循环，各case配置偏移`(repeat+case_index)%4`。全配置报告，不选择性删除失败；三重复不报P95、CI或显著性。

## 4 时间契约与正确性

所有四配置在计时内同等捕获8个time-embedding输入的GPU小clone，**不是额外模型调用**。计时结束后核验实际BF16归一化模型time与静态表逐值相同，raw scheduler timesteps也必须完全一致；不匹配即记失败，不解释质量结果。保留观察值和契约状态。

运行前验证：none模式复现R2；对角恒等；SHIFT/AFFINE不修改真实anchor及age；跨请求清空状态；修正Value dtype/device一致；无非有限表值。非零校正下只需验证当步fresh位置恒等，不要求整个局部输出等于dense。

F/R2每条输出PNG核验旧基线，分析另核repeat0的RGB；保存实际8次调用、活动数、cache年龄及输出hash。相同配置重复hash若不同，记录限制而非挑选。

## 5 计时与质量

端到端包含编码、cache初始化/索引、临时Value的FP32转换、查表、乘加、回cast、完整8步、scheduler、解码、cache清理与同步。排除下载、模型加载、静态表构建、PNG、指标；这些成本单列。阶段CUDA时间不强行相加为墙钟。

独立保存`runs_optimization.jsonl`、环境、CSV、summary和图像；不与primary合并重复。主参照为同期F，同case/repeat配对，取三次中位数及配对几何耗时比，报告allocated/reserved、真实cache和静态表字节。

质量取repeat0，沿用旧R0公共核心/band1/2/4和allowed-box远背景：LPIPS-mapmean≤0.02，且`MAE(候选,源图,far)−MAE(F,源图,far)≤1/255`。F是计算参考而非编辑真值；负extraBG不代表更优。人工非盲核对文字短语、手/发丝/衣领、盘面台阶；明显语义或边界失败可否决数值通过。

## 6 判定与下一步

逐例报告候选相对同期R2的LPIPS、公共band2/梯度误差及伪影变化，不只看全图均值。继续投入需至少2/3例LPIPS低于R2、原主要伪影有改善且无新明显退化；三例均通过原保真门槛，并相对F的跨例配对几何耗时比≤0.90。该规则只作探索筛查，不能称正式Go或无损；不得降低10%目标来迁就额外开销。

若质量改善但未达净延迟目标，记未达标/Revise；若SHIFT/AFFINE无收益，保留负结果，不宣布算法成功，也不据此否定所有上下文补偿。当前训练free规则只是候选，未完成新颖性核验。

后续先诊断layer-prefix或固定浅/中/深层刷新，严格计入获得当前hidden所需的上游传播，不把旧hidden重算当完整当前刷新。若仍有可校正残差，再预注册少样本监督残差头：限定因果可用特征、Train8/16/32嵌套划分、教师查询成本、同预算规则参照及包含校正器的完整耗时。没有部署时dense查询，不利用未来图像；RL继续暂缓。

当前状态：尚未启动，满足触发条件后执行；所有数值与质量结论待测。任何新增查询、模式或门槛改变均追加记录，不反写为已冻结方案。
