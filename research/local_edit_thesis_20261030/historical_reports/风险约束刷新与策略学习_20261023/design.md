# 风险约束刷新与策略学习：首轮冻结协议建议

2026-10-23。本文是新实验设计与纯策略模块合同；不将代码存在写成真实模型实验完成。GPU runner、来源清单、奖励工具版本及最终预登记由主任务另行冻结。旧速度头已失败，本轮不加载该头；以冻结B0执行器为共同基底，学习什么时候、在哪里做真实刷新。

## 1 六步序列决策与严格计算账本

采用原八步模型，0/7步强制完整目标计算，中间1—6步各作一次决策。强制核心C固定为输入提供的R2，不能为了满足预算裁切。文本与条件图始终更新，attention仍读取完整目标KV。

令背景目标数为Nbg，q=floor(Nbg/4)，余数r=Nbg mod4。余数E最多3个token，在全部中间步真实刷新；其余背景划为四个大小严格等于q的组G0—G3。实现复用2×2非核心tile的SHA排序，tile内按原坐标排序；先取余数，再按token切成四组，分界拆开的tile逐一记录。这是预算分组，**不是完整物体、结构闭合或自动语义边界分组**。初轮保留此定义以便精确控制计算量，距离环分组可作为以后单因素实验。

16个动作a=0…15分别是四组的任意子集，动作成本为popcount(a)个group-unit。中间六步总额严格为6units，因此每条轨迹实际选择目标总量相同：

`2*Ntarget + 6*(|C|+r) + 6*q`。

动作0只计算C∪E；动作15在该步计算全部目标，支付4units，是预算内完整回退。六步内不能无偿多次全图回退。每步采样前只保留满足`0 <= remaining-popcount(a) <= 4*remaining_steps_after`的动作，最后恰好用完预算。不可行动作直接拒绝；不得采样后静默替换，否则PPO的旧log_prob与执行动作不一致。

总token相同不等于端到端秒数相同，尤其不同步骤的形状、KV读取和固定条件图成本会改变延迟；真实时间必须另测。q=0时不启动本策略，标记无有意义的RL动作空间，按独立退化/完整路径协议处理。预算6units与所有中间读缓存年龄≤2一般不相容：四组在前、后三步各覆盖一次至少需8units。首轮记录年龄，不追加这个不可行硬界。

## 2 只能在执行前看到的状态

`build_features`固定输出54个可JSON序列化浮点数：全局10项、上一实际动作4个bit、四组各10项。包括步号、sigma、剩余额度与步骤数、核心/余数比例，当前latent相对真实锚点的漂移，真实锚点速度摘要，缓存年龄和固定源图ROI距离摘要。漂移与速度幅值用log1p压缩，不依赖Val统计。

决策发生在本步`base.forward`之前。此时可以读取当前z、最近真实计算时的z锚点、原始velocity缓存、`step-base.age`及源图ROI；不能读取本步尚未计算的A速度、fresh域外hidden、dense误差或未来终图。真实锚点只在对应token实际经过网络后更新；不能把策略估计或未计算的token标为fresh。

实现对CPU的base.age与GPU latent作显式设备转换；三个latent/velocity张量应在同设备。策略可部署在CPU，GPU统计到CPU的同步、特征构建、采样/argmax、索引传输都进入请求计时。54维中没有明确的提示词语义编码，不能称已实现语义风险预测。critic是回报估计，不是校准后的编辑失败概率。

## 3 监督预热、序列PPO与bandit的区别

共享小网络为54→64→64的Tanh MLP，输出16动作logits及一个critic值，共8,785参数。动作mask在Categorical分布构造前应用。底座不反传，采集时只保存状态、mask、动作、旧log_prob、旧value和回报；不保存完整KV作为训练样本，不展开八步底座梯度。

先在Train8上执行四条预冻结探索schedule：ROTATION=[1,2,4,8,1,2]、FRONT=[15,3,0,0,0,0]、BACK=[0,0,0,0,3,15]、固定seed随机可行动作。它们都用同一6units账本，不能事后挑预算。用训练专用完整F终图得到整条轨迹回报；每源选四条探索中最优整条轨迹，把其六个动作作为行为克隆标签。这个标签是“有限探索集合中的最佳完整轨迹”，不是真正的逐状态最优动作，也不是同状态分支的因果收益标签。STATIC仅在前三条deterministic schedule中选择Train8平均表现最好的共同schedule，不按Probe选；各源不同的RANDOM轨迹不能伪装成一条共同STATIC。

行为克隆只拟合上述动作，当前runner采用200次优化、Adam lr=1e-3；同时用终回报预训练critic。保存冻结IL权重，PPO从其副本开始。真实序列PPO按8轮×每轮8源执行64条完整轨迹：每个动作改变后续真实latent、缓存与年龄，最终回报经GAE分配给六步。当前方案采用Adam lr=3e-4、clip=0.1、每轮4个优化epoch、minibatch24、gamma=1、lambda=0.95、value系数0.5、entropy系数0.01、gradclip0.5、targetKL0.02；KL提前停止的优化步数照实记录，不增加rollout补足。

若策略仅在请求起点选择一条固定完整schedule，之后不观察状态，这属于contextual bandit/整轨迹选择；不能称为本设计的逐步序列RL。建议后续加入该对照；第一轮若PPO并未优于训练选出的STATIC或IL，就没有证据证明序列RL是必要的。

为排除“只是多看了训练结果”的解释，后续增强SL应取得与PPO相同的增量轨迹/回报，在固定预算下重做轨迹筛选行为克隆；这不需要额外教师前向，但新SL的优化、选择与评测仍需记账。当前冻结IL与PPO比较只隔离“原IL初始化上继续做PPO”的收益，不能单独证明PPO优于同数据增强SL。

## 4 训练终图奖励及质量约束

首轮使用同模型同初始latent的完整F作为训练专用终图参照，不在部署时运行教师。当前runner拟冻结如下终图代理回报：

`R = -(LPIPS/F + common_band2_MAE/F + 2*max(extraBG,0)) / .05`。

前五条transition奖励0，第六条为终回报；不增加在线dense或未计账的奖励模型。权重与归一化必须在rollout前冻结，不能边看Probe边调。固定计算预算下先优化质量，不把噪声单次延迟塞进首轮奖励后声称得到可靠时延最优。

F是模型输出而不是语义真值。LPIPS、额外背景和边界代理都不能保证指令正确；文字、接触关系、过度编辑仍需独立人工或明确工具审核。首轮不默认增加OCR/VLM奖励；若以后增加，先冻结工具与误差校验，并完整记录额外模型成本。不得在观测状态中放入teacher终图或误差。

B0仅在旧Train8中覆盖部分anchor0→后续步的系数。真实刷新后的未见anchor pair仍按原接口回退identity，不能伪称已校准所有年龄；记录pair覆盖和实际fallback，未来若重新拟合必须作为另一项训练成本。

## 5 主任务确认的一轮预算

| 环节 | 轨迹/请求 | 真实DiT上限 | 含义 |
|---|---:|---:|---|
| C5前置诊断 | 按主任务独立协议 | 112 | 同状态诊断，不能当部署成本 |
| Train8完整F缓存 | 最多8 | 64 | 若复用合法旧缓存，按实际新增量，不重复计费 |
| 四schedule探索 | 8源×4=32 | 256 | 训练专用终回报与IL标签 |
| PPO rollout | 8轮×8=64 | 512 | 只有轨迹查询计DiT，策略更新另计GPU/CPU时间 |
| Probe8七配置 | 8源×7=56 | 448 | F/B0/legacyB250/ROTATION/STATIC/IL/PPO |
| 统一预热 | 首个Train F同时承担 | 0额外 | 已包含在Train F缓存，不能再加8；不是每配置充分预热 |
| 小计 | 策略阶段160请求/1280DiT，另C5 | **1392** | 设计上限，不是已执行量 |
| 条件重复计时 | 主任务另行冻结 | ≤256 | 必须先有质量信号；不能用单次Probe秒数替代 |
| 条件总上限 | — | **≤1648** | 失败、重试及额外teacher/tool调用纳入总账 |

Probe8已经多次用于开发，首轮只支持机制和训练实现筛查，不是新留出泛化证明。若预算条件计时未通过质量前置，不花剩余额度自动搜索其他策略。策略优化只加载小网络，更新次数不折算成完整DiT。真实吞吐、模型加载、奖励计算、选择开销和峰值显存均实测，不从小头训练52秒外推PPO总时长。

共同等预算比较为ROTATION/STATIC/IL/PPO及其相同partition；legacyB250仍使用旧whole-tile cap，余数与逐步选择量可能不同，只作为额外Pareto对照，不能冒称严格同token。年龄规则在统一初值、每步选一组时可能与rotation完全相同，重合时注明重复轨迹，不把它算成额外独立证据。随机规则用于探索或显式随机对照，来源、seed与实际轨迹须保存。

## 6 继续、修订与停止

先通过纯策略CPU正确性和真实executor零策略/固定schedule复现，之后才训练。必须检验六步动作精确预算、C/E始终保留、首尾full、原坐标、真实年龄、actor输入不含本步未来量、环境采样与优化的mask/log_prob一致。正确性失败先停止，不用奖励训练掩盖。

当前runner拟冻结探索继续门：PPO相对ROTATION至少6/8源LPIPS降低≥1e-4，中位配对差<0，严格数值通过数不减少，公共band2及额外背景平均差不增加，没有新增明确语义失败，并且PPO也须优于IL。最终入口已冻结：PPO对IL使用与ROTATION完全相同的全部数值和语义谓词；extraBG使用来源平均配对差。同时完整报告STATIC和B250。数值门沿用LPIPS/F≤.02且extraBG≤1/255。只有全门通过才登记重复成本筛查；净节时目标≥10%，未达到则Revise，不称正式Go。

进入重复计时的门采用上述ROTATION与IL标准；主张RL必要性另需在同一组谓词下超过STATIC，并在后续共享PPO额外轨迹的监督控制上成立。若PPO奖励变好但上述终图/语义不改善，或仅胜过rotation而不胜过IL/STATIC，停止“RL带来必要增量”的主张；不得用训练回报、预算余量或较弱基线改写结论。32/64轨迹只是一轮有界试验，不能据一个策略训练seed宣布稳定。后续训练seed与新来源验证另行预登记，不在Probe上无限调超参数。

长期可采用`Train32_RL=原Train8+新Train24`，新增24源先按来源组去重并排除旧Probe8、所有Val及正式Test；同源多轮、裁剪、种子不可跨分区。若B0仍只由8源拟合，明确称“Train8校准器+Train32策略”，不能称全部模块32源重训。旧Val16已被观察，只是开发验证；需新独立Val时先冻结未见来源清单，Test64持续封存。正式结论前再做多策略训练seed、多生成seed、同预算增强SL与新留出比较。

## 7 模块接口与账本字段

`refresh_policy.py`提供`make_partition`、`BudgetController`、`build_features`、`ActorCritic`、`rule_action`、`supervised_update`、`ppo_update`。`check_refresh_policy.py`只使用CPU策略张量，不加载随机Qwen或真实底座。尚未在本机安装torch或运行真实模型；实际CPU验收回执由统一执行环境生成。

每个episode严格六条transition：`features:list[54]`、`mask:list[16]`、`action:int`、`log_prob:float`、`value:float`、`reward:float`、`done:bool`；最后一条done=True。PPO独立重放popcount总账；checkpoint保存feature名称、partition seed/schema、源码/数据/模型身份及CPU state_dict。加载相同ActorCritic后调用`load_state_dict`，部署固定argmax。没有可用动作必须显式报错，不临时变更预算。

runner在调用本步forward之前更新`base.indices[k]`与`indices_cpu[k]`，在真实网络计算之后更新锚点；请求结束清除策略history、真实anchor和执行器缓存。所有baseline共用模型、校准表、ROI与输入初始latent；F走原生完整路径。保存动作前后预算、所选原始indices、实际token、真实age、B0未见pair比例、六次策略调用、旧log_prob和checkpoint SHA。新增教师、全图回退与失败调用均显式入账。
