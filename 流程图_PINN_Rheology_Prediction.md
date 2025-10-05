# PINN流变性能预测系统 - 完整流程图

## 📊 整体架构流程

```mermaid
flowchart TD
    Start([开始]) --> Config[配置参数设置]
    Config --> DataLoad[数据加载]
    
    DataLoad --> DataProcess[数据预处理与特征工程]
    DataProcess --> DataSplit[数据集划分]
    DataSplit --> ModelDef[PINN模型定义]
    
    ModelDef --> Training[模型训练]
    Training --> Evaluation[模型评估]
    Evaluation --> Prediction[新样本预测]
    Prediction --> End([结束])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style Training fill:#87CEEB
    style Prediction fill:#DDA0DD
```

---

## 🔧 详细流程图

### 1️⃣ 配置与初始化阶段

```mermaid
flowchart LR
    A[系统配置] --> B[设备检测<br/>CUDA/CPU]
    B --> C[随机种子设置<br/>RANDOM_SEED=24]
    C --> D[超参数配置]
    
    D --> D1[BATCH_SIZE=64]
    D --> D2[LEARNING_RATE=8e-4]
    D --> D3[EPOCHS=1000]
    D --> D4[PHYSICS_WEIGHT=0.20]
    D --> D5[MC_SAMPLES=200]
    
    style A fill:#FFE4B5
    style D fill:#F0E68C
```

**关键参数说明：**
- 训练集：80% (~1499条)
- 验证集：20% (~375条)
- 最大训练轮数：1000轮
- 提前停止耐心值：50轮

---

### 2️⃣ 数据处理流程

```mermaid
flowchart TD
    A[读取PB_Data.csv<br/>1874条数据] --> B[基础特征工程]
    
    B --> B1[几何特征<br/>aspect_ratio = Length/Width]
    B --> B2[频率特征<br/>log_freq = log10_Freq]
    B --> B3[温度特征<br/>temp_inv = 1/T_K<br/>T_x_aspect]
    
    B --> C[分子量特征]
    C --> C1[PDI = Mw/Mn]
    C --> C2[log_Mw, log_Mn]
    
    B --> D[WLF时温等效]
    D --> D1[计算位移因子aT]
    D --> D2[reduced_freq = Freq × aT]
    
    B --> E[目标变量处理]
    E --> E1[tan_delta = G2/G1]
    E --> E2[数据权重设置<br/>outlier: weight=0]
    E --> E3[对数转换<br/>log_G1, log_G2]
    
    E3 --> F[数据标准化<br/>StandardScaler]
    F --> G[数据集划分]
    
    G --> G1[训练集 80%<br/>~1499条]
    G --> G2[验证集 20%<br/>~375条]
    
    G1 --> H[创建DataLoader]
    G2 --> H
    
    style A fill:#FFB6C1
    style F fill:#98FB98
    style G fill:#87CEEB
```

**特征列表（12个特征）：**
1. Length_nm
2. Width_nm
3. aspect_ratio
4. Temp_C
5. log_freq
6. temp_inv
7. T_x_aspect
8. aT
9. log_Mw
10. log_Mn
11. PDI
12. log_c

---

### 3️⃣ PINN模型架构

```mermaid
flowchart TD
    Input[输入特征 12维] --> Encoder[共享编码器]
    
    Encoder --> E1[Linear_256 + BatchNorm + GELU + Dropout]
    E1 --> E2[Linear_256 + BatchNorm + GELU + Dropout]
    E2 --> E3[Linear_256 + BatchNorm + GELU + Dropout]
    E3 --> E4[Linear_256 + BatchNorm + GELU + Dropout]
    
    E4 --> Split{分支处理}
    
    Split --> Branch1[G'预测分支]
    Branch1 --> B1_1[Linear_128 + BN + GELU + Dropout]
    B1_1 --> B1_2[Linear_64 + BN + GELU]
    B1_2 --> B1_3[Linear_1]
    B1_3 --> Out1[log_G' 输出]
    
    Split --> Branch2[G''预测分支]
    Branch2 --> B2_1[Linear_128 + BN + GELU + Dropout]
    B2_1 --> B2_2[Linear_64 + BN + GELU]
    B2_2 --> B2_3[Linear_1]
    B2_3 --> Out2[log_G'' 输出]
    
    Out1 --> Concat[拼接输出]
    Out2 --> Concat
    Concat --> Final[预测结果 2维]
    
    style Input fill:#FFE4B5
    style Encoder fill:#DDA0DD
    style Split fill:#FFB6C1
    style Final fill:#90EE90
```

**模型特点：**
- 深层共享编码器（4层×256神经元）
- 双分支结构（G' 和 G'' 独立预测）
- BatchNorm + GELU 激活函数
- Dropout (rate=0.05) 用于不确定性量化

---

### 4️⃣ 训练流程（核心）

```mermaid
flowchart TD
    Start([开始训练<br/>Epoch 1-1000]) --> Init[初始化优化器与调度器]
    
    Init --> Init1[AdamW优化器<br/>lr=8e-4, weight_decay=1e-4]
    Init --> Init2[OneCycleLR调度器]
    
    Init1 --> Loop{遍历训练批次}
    Init2 --> Loop
    
    Loop --> Forward[前向传播<br/>y_pred = model_X]
    
    Forward --> DataLoss[计算数据损失]
    DataLoss --> DL1[G'损失: MSE × 0.6]
    DataLoss --> DL2[G''损失: MSE × 0.4]
    DataLoss --> DL3[应用样本权重]
    
    Forward --> PhysicsLoss[计算物理约束损失]
    
    PhysicsLoss --> PL1[Cox-Merz规则<br/>复数粘度单调性<br/>权重×0.20]
    PhysicsLoss --> PL2[tan_δ_范围约束<br/>0 < tan_δ_ < 10<br/>权重×0.50]
    PhysicsLoss --> PL3[斜率范围约束<br/>-0.1 < slope < 0.4<br/>权重×0.40]
    PhysicsLoss --> PL4[平滑性约束<br/>二阶导数<br/>权重×0.20]
    PhysicsLoss --> PL5[热力学约束<br/>G', G'' > 0<br/>权重×0.05]
    
    DL3 --> Combine[组合总损失]
    PL1 --> Combine
    PL2 --> Combine
    PL3 --> Combine
    PL4 --> Combine
    PL5 --> Combine
    
    Combine --> TotalLoss[Total Loss = Data Loss<br/>+ physics_weight × Physics Loss]
    
    TotalLoss --> Backward[反向传播]
    Backward --> GradClip[梯度裁剪<br/>max_norm=1.0]
    GradClip --> Update[参数更新]
    
    Update --> MoreBatch{还有批次?}
    MoreBatch -->|是| Loop
    MoreBatch -->|否| Validation[验证阶段]
    
    Validation --> ValLoop{遍历验证批次}
    ValLoop --> ValForward[前向传播]
    ValForward --> ValLoss[计算验证损失]
    ValLoss --> MoreValBatch{还有批次?}
    MoreValBatch -->|是| ValLoop
    MoreValBatch -->|否| Compare[比较验证损失]
    
    Compare --> BestModel{验证损失<br/>是否改善?}
    BestModel -->|是| SaveModel[保存最佳模型<br/>重置patience=0]
    BestModel -->|否| IncPatience[patience + 1]
    
    SaveModel --> CheckEpoch
    IncPatience --> CheckStop{patience >= 50?}
    CheckStop -->|是| EarlyStop[提前停止]
    CheckStop -->|否| CheckEpoch
    
    CheckEpoch{Epoch < 1000?}
    CheckEpoch -->|是| Loop
    CheckEpoch -->|否| LoadBest[加载最佳模型]
    EarlyStop --> LoadBest
    
    LoadBest --> PlotHistory[绘制训练历史]
    PlotHistory --> End([训练完成])
    
    style Start fill:#90EE90
    style TotalLoss fill:#FFB6C1
    style PhysicsLoss fill:#DDA0DD
    style End fill:#87CEEB
```

**损失函数组成：**
1. **数据损失**：0.6×MSE(G') + 0.4×MSE(G'')
2. **物理约束损失**（5个组件）：
   - Cox-Merz规则 (20%)
   - tan(δ)范围 (50%)
   - 斜率范围 (40%)
   - 平滑性 (20%)
   - 热力学 (5%)

---

### 5️⃣ 模型评估流程

```mermaid
flowchart TD
    A[加载最佳模型] --> B[训练集评估]
    A --> C[验证集评估]
    
    B --> B1[前向传播]
    C --> C1[前向传播]
    
    B1 --> B2[反标准化]
    C1 --> C2[反标准化]
    
    B2 --> B3[计算评估指标]
    C2 --> C3[计算评估指标]
    
    B3 --> M1[MSE]
    B3 --> M2[MAE]
    B3 --> M3[R² Score]
    B3 --> M4[MAPE]
    
    C3 --> M5[MSE]
    C3 --> M6[MAE]
    C3 --> M7[R² Score]
    C3 --> M8[MAPE]
    
    M4 --> V1[可视化结果]
    M8 --> V1
    
    V1 --> V1_1[预测vs真实值散点图]
    V1 --> V1_2[残差分布图]
    V1 --> V1_3[不同频率段误差分析]
    V1 --> V1_4[样本级别预测曲线]
    
    style A fill:#FFE4B5
    style V1 fill:#98FB98
```

---

### 6️⃣ 预测与后处理流程

```mermaid
flowchart TD
    Start[新样本数据] --> FE[特征工程]
    
    FE --> FE1[几何特征]
    FE --> FE2[温度特征]
    FE --> FE3[WLF特征]
    FE --> FE4[分子量特征]
    
    FE1 --> Std[特征标准化]
    FE2 --> Std
    FE3 --> Std
    FE4 --> Std
    
    Std --> Pred{预测模式}
    
    Pred -->|带不确定性| MC[Monte Carlo Dropout]
    Pred -->|不带不确定性| Simple[简单预测]
    
    MC --> MC1[启用Dropout]
    MC1 --> MC2[重复预测200次]
    MC2 --> MC3[计算均值和标准差]
    MC3 --> MC4[构建置信区间<br/>95% CI]
    
    Simple --> S1[单次预测]
    
    MC4 --> Inverse[反标准化]
    S1 --> Inverse
    
    Inverse --> I1[对数空间 → 线性空间<br/>10^log_G]
    I1 --> I2[计算tan_δ_]
    
    I2 --> PostProcess[物理一致性后处理]
    
    PostProcess --> PP1[按频率排序]
    PP1 --> PP2[对数空间操作]
    
    PP2 --> PP3[高斯平滑<br/>σ=1.0]
    PP3 --> PP4[斜率裁剪<br/>[-0.1, 0.4]]
    PP4 --> PP5[迭代修正3次]
    
    PP5 --> PP6[tan_δ_约束<br/>clip_0, 10]
    PP6 --> PP7[从G'和tan_δ_<br/>重建G'']
    
    PP7 --> PP8[高频区域<br/>额外平滑]
    PP8 --> PP9[Cox-Merz规则<br/>修正]
    
    PP9 --> PP10{有不确定性?}
    PP10 -->|是| UC[更新置信区间]
    PP10 -->|否| Output
    
    UC --> UC1[保持相对不确定性]
    UC1 --> UC2[对边界平滑]
    UC2 --> UC3[确保边界包含预测值]
    
    UC3 --> Output[输出结果]
    
    Output --> Viz[可视化]
    Viz --> V1[G', G'' vs 频率]
    Viz --> V2[tan_δ_ vs 频率]
    Viz --> V3[Cole-Cole图]
    
    style Start fill:#FFE4B5
    style PostProcess fill:#DDA0DD
    style Output fill:#90EE90
```

**后处理关键步骤：**
1. **平滑处理**：高斯滤波去噪
2. **斜率约束**：限制变化速率
3. **物理约束**：tan(δ)范围
4. **一致性修正**：Cox-Merz规则
5. **不确定性传播**：保持置信区间

---

## 📈 数据流向图

```mermaid
flowchart LR
    Raw[原始数据<br/>1874条] --> |80/20划分| Train[训练集<br/>~1499条]
    Raw --> |80/20划分| Val[验证集<br/>~375条]
    
    Train --> |Batch 64| TrainLoader[训练DataLoader]
    Val --> |Batch 64| ValLoader[验证DataLoader]
    
    TrainLoader --> Model[PINN模型]
    ValLoader --> Model
    
    Model --> Pred1[log_G'标准化]
    Model --> Pred2[log_G''标准化]
    
    Pred1 --> |反标准化| G1_log[log_G']
    Pred2 --> |反标准化| G2_log[log_G'']
    
    G1_log --> |10^x| G1[G' Pa]
    G2_log --> |10^x| G2[G'' Pa]
    
    G1 --> |后处理| G1_final[G' 最终]
    G2 --> |后处理| G2_final[G'' 最终]
    
    style Raw fill:#FFB6C1
    style Model fill:#DDA0DD
    style G1_final fill:#90EE90
    style G2_final fill:#90EE90
```

---

## 🎯 核心创新点

```mermaid
mindmap
    root((PINN<br/>核心创新))
        物理约束
            Cox-Merz规则
            tan_δ_范围
            斜率约束
            平滑性
            热力学约束
        特征工程
            WLF时温等效
            分子量特征
            几何特征
            耦合特征
        不确定性量化
            MC Dropout
            200次采样
            置信区间
        后处理
            物理一致性
            高斯平滑
            迭代修正
            边界传播
```

---

## 📊 性能监控流程

```mermaid
flowchart TD
    A[每个Epoch] --> B[记录训练损失]
    B --> B1[总损失]
    B --> B2[数据损失]
    B --> B3[物理损失]
    B --> B4[各组件损失]
    
    A --> C[记录验证损失]
    C --> C1[总损失]
    C --> C2[物理损失]
    
    B4 --> D[每10轮打印]
    C2 --> D
    
    D --> E[训练完成后]
    E --> F[绘制损失曲线]
    F --> F1[总损失对比]
    F --> F2[物理约束损失]
    F --> F3[各组件趋势]
    
    C1 --> G{验证损失比较}
    G -->|改善| H[保存最佳模型]
    G -->|未改善| I[patience计数]
    I --> J{patience >= 50?}
    J -->|是| K[提前停止]
    J -->|否| Continue[继续训练]
    
    style A fill:#FFE4B5
    style K fill:#FFB6C1
    style H fill:#90EE90
```

---

## 🔍 完整系统总结

| 模块 | 输入 | 输出 | 关键技术 |
|------|------|------|----------|
| **数据处理** | PB_Data.csv | 标准化特征矩阵 | 特征工程、WLF、标准化 |
| **模型架构** | 12维特征 | 2维预测(log_G', log_G'') | 深层编码器、双分支 |
| **训练** | 训练/验证集 | 最佳模型权重 | 物理约束、动态权重、提前停止 |
| **预测** | 新样本 | G', G'', 不确定性 | MC Dropout、后处理 |
| **可视化** | 预测结果 | 多种图表 | 散点图、Cole-Cole图 |

**系统优势：**
✅ 结合物理知识和数据驱动  
✅ 提供预测不确定性  
✅ 自动物理一致性修正  
✅ 完整的训练监控  
✅ 稳健的优化策略  

**实际表现：**
- 训练集：~1499条
- 验证集：~375条
- 训练轮数：1000轮（完整）
- 收敛情况：持续改进，未提前停止
