# House Prices V4：在组员 V3 上继续改进

## 实测结论

V4 已训练、生成预测并保存 notebook 执行结果。两组相同数据划分下均优于本机复现的 V3；**尚未达到 0.110**。

用户实际提交后的 Kaggle 成绩：V3 **0.11881**，V4 **0.11735**。下表是本地交叉验证成绩，与 Kaggle 成绩分别记录。

| 五折随机种子 | 本机复现 V3 | V4 | 误差下降 |
|---|---:|---:|---:|
| 42 | 0.117905487 | 0.114678905 | 2.74% |
| 2026 | 0.115709007 | 0.113709185 | 1.73% |

10 个折对照中 9 个误差下降。全部 1,460 行验证房屋保留，没有通过移除难样本降低分数。
组员 V3 notebook 保存的结果是约 0.117973，本机复现为 0.117905487；软件/平台差异可能导致小幅数值差异，本表统一使用本机成对复跑的结果。

## V4 具体改动

最终使用 4 个分支、3 类算法，按对数房价融合：

- ContextSVR：47.4925%。训练折内按特征与房价的绝对相关系数调整核距离；增加 Neighborhood、Neighborhood×SaleCondition、Neighborhood×OverallQual 三个平滑留一目标统计。SVR C=3、gamma=.0003、epsilon=.02。
- ClippedCatBoost：23.4833%。沿用 V3 CatBoost 参数，对连续数值特征使用训练折内 99% 上限，限制极端值影响。
- NeighborhoodElasticNet：22.8435%。增加训练折内街区相对面积、质量与面积交互，使用 RobustScaler、97.5% 上限，alpha=.0005、l1_ratio=.9。
- NeighborhoodSVR：6.1807%。同类几何表示、StandardScaler、99% 上限，作为少量互补分支。

上述上限只改变特征值，不删房屋，也不调整真实房价。Id 只用于输出和核对，不作为模型特征或修价条件。最终全量训练 1,460 行，输出全部 1,459 条测试预测。

## 验证含义与边界

沿用组员 V3 的五折 OOF 开发选择方式。seed42 用于选特征、参数和非负权重，固定后在 seed2026 上检查稳定性。目标统计中的训练行原始组均值及全局先验均排除自身售价；验证编码只查训练组统计。所有其他填补、编码、分位数、缩放和距离权重也只在当前训练折学习。

本轮共记录 513 个模型/预处理配置，包含未奏效方向；它们不是最终融合的 513 个模型。大范围选择会使开发成绩乐观。第二 seed 也不是从未看过的独立测试集，不能据此保证 Kaggle 或新数据成绩。没有使用外部测试真实售价。脚本只导出预测文件，Kaggle 提交由用户完成。

## 如何看与运行

1. 在 VS Code 中打开 house_prices_v4.ipynb。结果和 4 张图已保存，可直接阅读。
2. notebook、house_prices_v4_pipeline.py、house_prices_v4_core.py 放在同一目录。
3. 默认识别仓库中的 Dataset/train.csv 和 Dataset/test.csv，也兼容 house-prices-advanced-regression-techniques/、Proj/Dataset/ 与 Kaggle 官方数据目录。
4. 使用 Python 3.12–3.14；本次实测为 Python 3.14.5。requirements_v4.txt 中的固定版本要求 Python 至少为 3.12。在新环境安装这些依赖，不需要复制本机临时目录。Mac 上 XGBoost 若提示缺少 libomp，可安装 OpenMP：`brew install libomp`，参见 https://xgboost.readthedocs.io/en/stable/install.html 。
5. 默认 FORCE_RETRAIN=False 会核验数据、源码、配置、包版本和输出哈希后读取缓存。改为 True 会重做最终方案验证与训练，而不是重跑整个 513 配置搜索。更换数据或环境后请用 True。

首次安装（在仓库根目录执行；将 python3 换为已安装的 Python 3.12–3.14）：

```bash
python3 -m venv .venv-v4
source .venv-v4/bin/activate
python -m pip install -r requirements_v4.txt
```

Windows 激活命令为 `.venv-v4\Scripts\activate`。在 VS Code 中打开仓库文件夹，安装 Python 和 Jupyter 扩展，然后为 notebook 选择此虚拟环境的 Python 内核。

运行已冻结的最终方案：

```bash
python house_prices_v4_pipeline.py --force
```

V3 文件没有修改。所用 V3 特征/预处理参考实现已保留在 core 中，因此复跑 V4 不依赖更早的个人 V2。

## 文件

- house_prices_v4.ipynb：已执行的结果说明、对照、图表。
- house_prices_v4_pipeline.py：冻结方案、目标统计、训练、验证、预测与完整性检查。
- house_prices_v4_core.py：独立环境加载、组员 V3 参考特征/预处理及基线复现。
- requirements_v4.txt：实际使用的包版本。
- submission_v4.csv：最终 1,459 条预测。
- output/project/v4/final/：运行后生成两 seed OOF、分数、权重、输入/源码/输出哈希和验证记录。
- output/project/v4/search_*：本机保留的完整探索记录，不包含在这次运行文件补充中；最终方案脚本不会重新执行全部搜索。

仓库的 notebook 已保存执行结果，可直接阅读。首次完整运行会训练模型并生成输出；后续匹配的运行可以读取已核验的缓存。运行文件补充未上传本机虚拟环境、临时依赖或 OpenMP 二进制文件。
