# CT GNSS 协方差尺度自适应

该选项用于 `GINS_CT` 的批量和滑窗路径，默认关闭。前端 GNSS 解算、门控、原始位置协方差、样条节点和 IMU 模型保持各自职责；LC/TC 未接入此选项。

在 XML 的 `<fgo>` 中启用：

```xml
<ct_gnss_adaptive>1</ct_gnss_adaptive>
<ct_gnss_scale_lambda>1</ct_gnss_scale_lambda>
<ct_gnss_scale_max>100</ct_gnss_scale_max>
```

`ct_gnss_adaptive` 只接受 0 或 1，缺省为 0。启用时要求 lambda 有限且不小于 0，scale_max 有限且不小于 1。缺省 lambda=1、scale_max=100 是原型参数，尚未通过跨数据集调参或验证确定。

## 观测与尺度

`CtGnssPosition::position_covariance_ecef_m2` 始终保存原始 ECEF 位置协方差 R，单位 m²；`covariance_scale` 保存无量纲尺度 a，初始为 1。有效协方差为 aR，因子中的原始平方根信息矩阵乘以 1/sqrt(a)。a=4 表示标准差变为原来的两倍，信息权重降为四分之一。

初解给出的 GNSS 天线残差为：

```text
e_k = p_IMU(t_k) + C_body_to_ECEF(t_k) * lever_body - z_GNSS,k
q_k = e_k^T * R_k^-1 * e_k
```

残差计算使用与因子相同的固定 ECEF 原点和机体系 IMU→天线杆臂，q 始终使用原始 R，而非上一轮的 aR。该 q 是优化后的拟合残差统计量，包含轨迹拟合影响；不是独立的 GNSS 创新检验，也不是严格 NIS。

## 尺度更新

按观测时间顺序，在 1≤a_k≤a_max 内最小化：

```text
q_k/a_k + 3*log(a_k) + lambda*(log(a_k)-log(a_{k-1}))²
```

三维位置高斯模型的 log-det 项给出 3*log(a_k)。令 s=log(a)，导数为 `-q*exp(-s)+3+2*lambda*(s-log(a_prev))`，二阶导数非负。先判断边界，再二分求内部唯一驻点。lambda=0 时结果为 clamp(q/3,1,a_max)。

每次窗口求解依次执行：

1. 使用当前已接受的尺度完成轨迹初解。
2. 对本轮实际参与求解的 GNSS 观测，按时间顺序更新一次尺度；上一条已经更新的尺度作为当前时间先验。首条取 1；滑窗中若保留了窗前观测，则用它的已接受尺度作为边界。
3. 固定尺度，再求解一次轨迹。
4. 两轮均得到可用解才提交轨迹、零偏和尺度；失败保留原状态。

这是一次尺度扫描加两轮轨迹求解的交替更新原型，没有将所有尺度与轨迹联合迭代到 MAP 收敛。时间正则按相邻观测定义，不根据失锁时长自动衰减。重叠窗口可以重估仍活动的尺度，但不会把乘数累计到原始协方差上。

## 边缘化和日志

出窗线性化与求解共用 `CtGnssPositionFactor` 和已接受的尺度。出窗观测一旦进入先验，其权重不再重估。尺度更新的 log-det 和时间项在固定尺度后不依赖轨迹，不作为额外轨迹因子加入先验。

启用后输出：

- `[CT GNSS SCALE]`：窗口编号、初解迭代次数和两轮总迭代次数。
- `[CT GNSS WEIGHT]`：窗口编号、观测相对处理起点的秒数、初解 q、上一历元尺度和当前尺度。重叠窗口对同一观测有多条记录。

原有 `[CT WINDOW]` 中的 Ceres 报告、initial_cost 和 final_cost 只描述最后一轮。两轮使用不同权重，不能将第一轮初始代价与第二轮最终代价直接比较并宣称代价改善。

初解只要求产生可用解，最终 `converged` 标记描述最后一轮。新增参数在读取观测、创建导航对象和启动绘图线程之前校验，非法值输出 `[CT ERROR]` 并返回退出码 1。后端 API 保留自己的参数校验，供直接调用。其他既有 CT 错误路径未在此修改，自动检查仍应同时查看错误日志和有效输出。

## 验证口径

测试包含尺度边界与驻点、lambda=0 的解析解、完整协方差白化及雅可比缩放、两轮求解、原始协方差不变、反复求解不重复放大、失败回滚，以及出窗 GNSS Hessian 的信息比例。真实数据对照需同时记录轨迹精度、尺度分布、求解耗时和 GNSS 前端结果是否一致。

本机本轮使用以下构建方式，关闭 MSBuild 节点复用，并用 `/Z7` 避开 PDB 管理器不匹配；环境变量仅作用于当前终端进程。新增含中文的头文件与同目录文件一样使用 UTF-8 BOM。

```powershell
$env:_CL_='/Z7'
cmake --build build --config RelWithDebInfo --target GREAT_GINSFGO -- /m:1 /nr:false
cmake --build build/ct_backend_solve_check --config RelWithDebInfo -- /m:1 /nr:false
ctest --test-dir build/ct_backend_solve_check -C RelWithDebInfo --output-on-failure
& 'E:\SoftWare\Miniconda3\envs\modeling\python.exe' -B plot/check_ct_gnss_adaptive.py
```

真实数据检查脚本要求本轮预实验基准和修改前运行库快照存在，拒绝覆盖已有结果目录。最终结果、配置、尺度日志和权重最优性复核保存在 `results/ct_gnss_adaptive_final_20261029/`。`tests/` 和 `results/` 沿用仓库的忽略规则，分享改动时需要另行携带这些验证文件。

入口前置校验后的补充验证在该结果目录的 `closeout/`：六种非法配置必须在读取观测前返回 1；LC、CT 批量、CT 滑窗三个 30 s 回归须与修改前的 INS/FLT 逐字节一致；另有启用自适应的 30 s 运行和尺度最优性检查。完整算例的算法 DLL 与收尾程序通过哈希核对相同，入口修改后的可执行文件版本单独保存在 `closeout/runtime.json`。

首轮校园数据完成输出后发生访问冲突，已保留在 `results/ct_gnss_adaptive_20261029/`。最小可视化生命周期用例确认，viewer 析构后绘图线程仍可能访问其互斥量和轨迹。修复为析构前调用 Hide() 停止并 join 绘图线程，停止标志采用原子变量；该修复不改导航残差或权重。修复前后的已释放内存保护用例及日志在 `build/ct_gnss_adaptive_20261029/debugger/`。

该模型只能修正置信度，不能消除有偏坐标、错误杆臂或时间同步偏差。精度收益和与已有方法的区别需另行实验与文献对比。
