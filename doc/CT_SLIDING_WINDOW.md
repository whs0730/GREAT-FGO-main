# 连续时间 GNSS/INS 滑窗

`GINS_CT` 独立路径已支持窗口推进及 Schur 边缘化，保留原批量入口。CT 边缘化在 `_gins_marginalization()` 的独立分支中执行，原 LC/TC 分支保留。实测精度对比仍按用户要求留到最后。

## 配置

在原 CT XML 的 `<fgo>` 内增加：

```xml
<ct_window_sec>10</ct_window_sec>
<ct_update_sec>1</ct_update_sec>
<ct_knot_sec>1</ct_knot_sec>
```

| 字段 | 单位与含义 | 未配置时 |
| --- | --- | --- |
| `ct_window_sec` | 窗口时长，秒；必须至少为两个节点间隔 | 0，继续整段批量求解 |
| `ct_update_sec` | 求解更新周期，秒；不小于 IMU 输入间隔，不大于窗口时长 | 1 s |
| `ct_knot_sec` | 位置和姿态共用的样条节点间隔，秒 | 0，沿用 GNSS 采样间隔 |

`<ins out_freq="..." int_sec="...">` 仍控制 INS 输出频率，与求解更新周期相互独立。窗口边界对齐样条节点，因此实际保留时长可比配置多不足一个节点间隔；控制点还包含三次样条所需的边界支撑点。更改节点间隔不是本阶段的精度调参结论。

当前仍要求同一 GPS 周内正向处理、`GNSS Type="LCI"`，使用机体系 IMU→天线杆臂、固定 ECEF 重力和每窗常值 IMU 零偏。

## 窗口如何推进

1. 复用已有配置姿态或辅助对准，持续读入解码后的 IMU 及前端 GNSS 位置。首窗达到时长后初始化轨迹；不足一窗的短文件在结束时初始化并求解。
2. 后续保留重叠区的位置、姿态控制点及当前零偏估计。边缘化模式从上一窗最后观测时刻的样条位置、速度、加速度和机体系角速度生成末端新控制点初值，避免反复外推未被观测约束的尾点。位置初值包含三次样条的 `-a*dt²/6` 控制点修正；这些预测只用于初值，不作为观测或历史先验。
3. 左边界按整数节点推进，删除旧控制点、过期 IMU 和 GNSS。额外保留一条左边界之前的 GNSS，仅用于维持输入流的时间顺序检查；该观测不会再次进入窗口残差。
4. 删除旧控制点之前，主流程调用 `_gins_marginalization()` 的 CT 分支：只线性化将离开窗口的原始观测因子、上一窗边缘化先验，以及尚未吸收的初始姿态先验；用 Schur 补消去旧控制点，形成保留控制点及常值零偏的先验。下一窗不再固定前三个边界控制点，重叠区域可以继续调整。
5. 新先验先作为待提交结果保存；窗口准备、裁剪及求解在副本上进行，只有可用解才同时提交轨迹和新先验。失败时保留旧先验、旧轨迹及待处理观测；主程序报告 CT 错误并停止，不把失败窗口作为新解输出。
6. 首窗输出从共同观测起点开始的状态；后续每次只输出尚未写过的历元，已输出状态不回写。末尾不足更新周期的数据也会求解和输出，不重复写同一终点。

窗口状态接口仍为 `GetState(t)`，有效范围是当前样条的 `[MinTime(), MaxTime())`。旧状态离开窗口后不可再查询，历史轨迹应从已经写出的 INS 文件读取。新增观测后，窗口模式保留旧轨迹可查询，但 `IsSolved()` 为 false，直到窗口更新被接受。

GNSS 暂时中断且已有历史先验时，可以使用 IMU 继续推进；首次初始化仍需要共同区间内至少两条 GNSS 和两条 IMU。因子可装配不等于所有运动条件下都可观测，也不构成中断精度保证。

## 与老师 GINS 的对应

- 参考老师 `LibFGOMSF/gfactor/marginalization_factor.cpp` 的 Ceres 切空间线性化、`[旧变量 | 保留变量]` 排列、Schur 伪逆及特征值平方根先验；对应的 CTraj 版权和 BSD 声明保留在新增头文件中。
- 老师保存参数地址并在扩展后重映射；当前实现保存控制点索引、线性化状态和矩阵，由每次 Ceres Problem 重新获取地址，适应后端副本求解和队首删除。
- 老师使用 `b=-Jᵀr`、先验 `J*dx-r0`；当前使用梯度 `g=Jᵀr`、先验 `r0+J*dx`，两种符号约定等价。伪逆与平方根分解采用与参考代码相同的 `1e-8` 特征值阈值。
- 四元数局部差使用与 `EigenQuaternionManifold::Minus` 一致的左乘半角坐标，先验因子通过自动微分计算 Jacobian。3 维位置和两组零偏使用向量差。
- 仍保留窗口内的原始重叠观测，因此只吸收将退出窗口的因子，避免重复计数。标定零偏先验不压入历史先验，每个窗口仍只施加一份；没有引入老师的随机游走零偏滤波或人工末端约束。

## 这一版的边界

- 历史信息通过线性化的 Schur 先验传递，存在非线性重线性化误差；不能视为与整段非线性批量优化完全等价。
- 主流程启用滑窗时自动调用 CT 边缘化分支。直接使用后端 API 时，应先 `EnableMarginalization(true)`，每次移动前通过 `LinearizeMarginalization()`、Schur 消元及 `SetPendingMarginalization()` 提供先验；未启用的独立后端仍保留固定三个边界控制点的旧接口行为。
- 控制点、CT 观测缓冲和输出质量记录随窗口裁剪。原输入解码器仍会预读数据文件，不能据此声称整个进程的内存与文件长度无关。
- 不包含在线网络输入、跨周处理、时变零偏、重力估计、轨迹协方差或 CT KML。
- `ct_window_sec=0` 保留批量行为；窗口输出具有较短的数据使用范围，与原整段平滑结果不必相同。

## 复现入口

后端检查使用 `tests/ct_gins_backend`，包括原联合求解、固定边界窗口回归，以及新增 `ct_backend_marginal_check`。边缘化检查覆盖 Ceres Minus、四元数符号不变性、切空间有限差分、QR 消元的正规方程核验、先验传递、GNSS 中断、大时间戳、无重复计数及失败回滚。这是合成正确性检查，不是实测精度评定。

```powershell
cmake -S tests/ct_gins_backend -B build/ct_backend_solve_check -G 'Visual Studio 18 2026' -A x64
cmake --build build/ct_backend_solve_check --config RelWithDebInfo --parallel 1 -- /nr:false
ctest --test-dir build/ct_backend_solve_check -C RelWithDebInfo --output-on-failure
cmake --build build --config RelWithDebInfo --target GREAT_GINSFGO --parallel 1 -- /nr:false
```

本次边缘化功能验证配置、日志及 `verification.json` 保存在 `build/ct_marginal_check/`；原固定边界阶段证据保留在 `build/ct_window_check/`。这些目录和当前 `tests/` 均受 Git 忽略，换机器需单独保存。功能检查脚本不读取真值、不计算 RMSE，检查原 LC/批量输出回归、滑窗缓存、CT 分支先验调用、观测计数、输出间隔及质量标签。
