# PINN Rheology Prediction System - Complete Flowchart

## 📊 Overall Architecture Flow

```mermaid
flowchart TD
    Start([Start]) --> Config[Configuration Setup]
    Config --> DataLoad[Data Loading]
    
    DataLoad --> DataProcess[Data Preprocessing & Feature Engineering]
    DataProcess --> DataSplit[Dataset Split]
    DataSplit --> ModelDef[PINN Model Definition]
    
    ModelDef --> Training[Model Training]
    Training --> Evaluation[Model Evaluation]
    Evaluation --> Prediction[New Sample Prediction]
    Prediction --> End([End])
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style Training fill:#87CEEB
    style Prediction fill:#DDA0DD
```

---

## 🔧 Detailed Flowcharts

### 1️⃣ Configuration & Initialization Phase

```mermaid
flowchart LR
    A[System Configuration] --> B[Device Detection<br/>CUDA/CPU]
    B --> C[Random Seed Setup<br/>RANDOM_SEED=24]
    C --> D[Hyperparameter Config]
    
    D --> D1[BATCH_SIZE=64]
    D --> D2[LEARNING_RATE=8e-4]
    D --> D3[EPOCHS=1000]
    D --> D4[PHYSICS_WEIGHT=0.20]
    D --> D5[MC_SAMPLES=200]
    
    style A fill:#FFE4B5
    style D fill:#F0E68C
```

**Key Parameters:**
- Training set: 80% (~1499 samples)
- Validation set: 20% (~375 samples)
- Max training epochs: 1000
- Early stopping patience: 50 epochs

---

### 2️⃣ Data Processing Flow

```mermaid
flowchart TD
    A[Load PB_Data.csv<br/>1874 samples] --> B[Basic Feature Engineering]
    
    B --> B1[Geometric Features<br/>aspect_ratio = Length/Width]
    B --> B2[Frequency Features<br/>log_freq = log10_Freq]
    B --> B3[Temperature Features<br/>temp_inv = 1/T_K<br/>T_x_aspect]
    
    B --> C[Molecular Weight Features]
    C --> C1[PDI = Mw/Mn]
    C --> C2[log_Mw, log_Mn]
    
    B --> D[WLF Time-Temperature Superposition]
    D --> D1[Calculate Shift Factor aT]
    D --> D2[reduced_freq = Freq × aT]
    
    B --> E[Target Variable Processing]
    E --> E1[tan_delta = G2/G1]
    E --> E2[Data Weight Setup<br/>outlier: weight=0]
    E --> E3[Log Transformation<br/>log_G1, log_G2]
    
    E3 --> F[Data Standardization<br/>StandardScaler]
    F --> G[Dataset Split]
    
    G --> G1[Training Set 80%<br/>~1499 samples]
    G --> G2[Validation Set 20%<br/>~375 samples]
    
    G1 --> H[Create DataLoader]
    G2 --> H
    
    style A fill:#FFB6C1
    style F fill:#98FB98
    style G fill:#87CEEB
```

**Feature List (12 features):**
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

### 3️⃣ PINN Model Architecture

```mermaid
flowchart TD
    Input[Input Features 12D] --> Encoder[Shared Encoder]
    
    Encoder --> E1[Linear_256 + BatchNorm + GELU + Dropout]
    E1 --> E2[Linear_256 + BatchNorm + GELU + Dropout]
    E2 --> E3[Linear_256 + BatchNorm + GELU + Dropout]
    E3 --> E4[Linear_256 + BatchNorm + GELU + Dropout]
    
    E4 --> Split{Branch Split}
    
    Split --> Branch1[G' Prediction Branch]
    Branch1 --> B1_1[Linear_128 + BN + GELU + Dropout]
    B1_1 --> B1_2[Linear_64 + BN + GELU]
    B1_2 --> B1_3[Linear_1]
    B1_3 --> Out1[log_G' Output]
    
    Split --> Branch2[G'' Prediction Branch]
    Branch2 --> B2_1[Linear_128 + BN + GELU + Dropout]
    B2_1 --> B2_2[Linear_64 + BN + GELU]
    B2_2 --> B2_3[Linear_1]
    B2_3 --> Out2[log_G'' Output]
    
    Out1 --> Concat[Concatenate Output]
    Out2 --> Concat
    Concat --> Final[Prediction Result 2D]
    
    style Input fill:#FFE4B5
    style Encoder fill:#DDA0DD
    style Split fill:#FFB6C1
    style Final fill:#90EE90
```

**Model Features:**
- Deep shared encoder (4 layers × 256 neurons)
- Dual-branch architecture (G' and G'' independent prediction)
- BatchNorm + GELU activation function
- Dropout (rate=0.05) for uncertainty quantification

---

### 4️⃣ Training Flow (Core)

```mermaid
flowchart TD
    Start([Start Training<br/>Epoch 1-1000]) --> Init[Initialize Optimizer & Scheduler]
    
    Init --> Init1[AdamW Optimizer<br/>lr=8e-4, weight_decay=1e-4]
    Init --> Init2[OneCycleLR Scheduler]
    
    Init1 --> Loop{Iterate Training Batches}
    Init2 --> Loop
    
    Loop --> Forward[Forward Pass<br/>y_pred = model_X]
    
    Forward --> DataLoss[Calculate Data Loss]
    DataLoss --> DL1[G' Loss: MSE × 0.6]
    DataLoss --> DL2[G'' Loss: MSE × 0.4]
    DataLoss --> DL3[Apply Sample Weights]
    
    Forward --> PhysicsLoss[Calculate Physics Constraint Loss]
    
    PhysicsLoss --> PL1[Cox-Merz Rule<br/>Complex Viscosity Monotonicity<br/>Weight×0.20]
    PhysicsLoss --> PL2[tan_δ_ Range Constraint<br/>0 < tan_δ_ < 10<br/>Weight×0.50]
    PhysicsLoss --> PL3[Slope Range Constraint<br/>-0.1 < slope < 0.4<br/>Weight×0.40]
    PhysicsLoss --> PL4[Smoothness Constraint<br/>Second Derivative<br/>Weight×0.20]
    PhysicsLoss --> PL5[Thermodynamic Constraint<br/>G', G'' > 0<br/>Weight×0.05]
    
    DL3 --> Combine[Combine Total Loss]
    PL1 --> Combine
    PL2 --> Combine
    PL3 --> Combine
    PL4 --> Combine
    PL5 --> Combine
    
    Combine --> TotalLoss[Total Loss = Data Loss<br/>+ physics_weight × Physics Loss]
    
    TotalLoss --> Backward[Backward Pass]
    Backward --> GradClip[Gradient Clipping<br/>max_norm=1.0]
    GradClip --> Update[Parameter Update]
    
    Update --> MoreBatch{More Batches?}
    MoreBatch -->|Yes| Loop
    MoreBatch -->|No| Validation[Validation Phase]
    
    Validation --> ValLoop{Iterate Validation Batches}
    ValLoop --> ValForward[Forward Pass]
    ValForward --> ValLoss[Calculate Validation Loss]
    ValLoss --> MoreValBatch{More Batches?}
    MoreValBatch -->|Yes| ValLoop
    MoreValBatch -->|No| Compare[Compare Validation Loss]
    
    Compare --> BestModel{Val Loss<br/>Improved?}
    BestModel -->|Yes| SaveModel[Save Best Model<br/>Reset patience=0]
    BestModel -->|No| IncPatience[patience + 1]
    
    SaveModel --> CheckEpoch
    IncPatience --> CheckStop{patience >= 50?}
    CheckStop -->|Yes| EarlyStop[Early Stopping]
    CheckStop -->|No| CheckEpoch
    
    CheckEpoch{Epoch < 1000?}
    CheckEpoch -->|Yes| Loop
    CheckEpoch -->|No| LoadBest[Load Best Model]
    EarlyStop --> LoadBest
    
    LoadBest --> PlotHistory[Plot Training History]
    PlotHistory --> End([Training Complete])
    
    style Start fill:#90EE90
    style TotalLoss fill:#FFB6C1
    style PhysicsLoss fill:#DDA0DD
    style End fill:#87CEEB
```

**Loss Function Components:**
1. **Data Loss**: 0.6×MSE(G') + 0.4×MSE(G'')
2. **Physics Constraint Loss** (5 components):
   - Cox-Merz Rule (20%)
   - tan(δ) Range (50%)
   - Slope Range (40%)
   - Smoothness (20%)
   - Thermodynamics (5%)

---

### 5️⃣ Model Evaluation Flow

```mermaid
flowchart TD
    A[Load Best Model] --> B[Training Set Evaluation]
    A --> C[Validation Set Evaluation]
    
    B --> B1[Forward Pass]
    C --> C1[Forward Pass]
    
    B1 --> B2[Inverse Transform]
    C1 --> C2[Inverse Transform]
    
    B2 --> B3[Calculate Evaluation Metrics]
    C2 --> C3[Calculate Evaluation Metrics]
    
    B3 --> M1[MSE]
    B3 --> M2[MAE]
    B3 --> M3[R² Score]
    B3 --> M4[MAPE]
    
    C3 --> M5[MSE]
    C3 --> M6[MAE]
    C3 --> M7[R² Score]
    C3 --> M8[MAPE]
    
    M4 --> V1[Visualization Results]
    M8 --> V1
    
    V1 --> V1_1[Prediction vs True Scatter Plot]
    V1 --> V1_2[Residual Distribution]
    V1 --> V1_3[Error Analysis by Frequency Range]
    V1 --> V1_4[Sample-level Prediction Curves]
    
    style A fill:#FFE4B5
    style V1 fill:#98FB98
```

---

### 6️⃣ Prediction & Post-processing Flow

```mermaid
flowchart TD
    Start[New Sample Data] --> FE[Feature Engineering]
    
    FE --> FE1[Geometric Features]
    FE --> FE2[Temperature Features]
    FE --> FE3[WLF Features]
    FE --> FE4[Molecular Weight Features]
    
    FE1 --> Std[Feature Standardization]
    FE2 --> Std
    FE3 --> Std
    FE4 --> Std
    
    Std --> Pred{Prediction Mode}
    
    Pred -->|With Uncertainty| MC[Monte Carlo Dropout]
    Pred -->|Without Uncertainty| Simple[Simple Prediction]
    
    MC --> MC1[Enable Dropout]
    MC1 --> MC2[Repeat Prediction 200 Times]
    MC2 --> MC3[Calculate Mean & Std]
    MC3 --> MC4[Build Confidence Interval<br/>95% CI]
    
    Simple --> S1[Single Prediction]
    
    MC4 --> Inverse[Inverse Transform]
    S1 --> Inverse
    
    Inverse --> I1[Log Space → Linear Space<br/>10^log_G]
    I1 --> I2[Calculate tan_δ_]
    
    I2 --> PostProcess[Physical Consistency Post-processing]
    
    PostProcess --> PP1[Sort by Frequency]
    PP1 --> PP2[Log Space Operation]
    
    PP2 --> PP3[Gaussian Smoothing<br/>σ=1.0]
    PP3 --> PP4[Slope Clipping<br/>[-0.1, 0.4]]
    PP4 --> PP5[Iterative Correction 3x]
    
    PP5 --> PP6[tan_δ_ Constraint<br/>clip_0, 10]
    PP6 --> PP7[Reconstruct G'' from<br/>G' and tan_δ_]
    
    PP7 --> PP8[High Frequency Region<br/>Extra Smoothing]
    PP8 --> PP9[Cox-Merz Rule<br/>Correction]
    
    PP9 --> PP10{Has Uncertainty?}
    PP10 -->|Yes| UC[Update Confidence Interval]
    PP10 -->|No| Output
    
    UC --> UC1[Maintain Relative Uncertainty]
    UC1 --> UC2[Smooth Boundaries]
    UC2 --> UC3[Ensure Bounds Contain Prediction]
    
    UC3 --> Output[Output Results]
    
    Output --> Viz[Visualization]
    Viz --> V1[G', G'' vs Frequency]
    Viz --> V2[tan_δ_ vs Frequency]
    Viz --> V3[Cole-Cole Plot]
    
    style Start fill:#FFE4B5
    style PostProcess fill:#DDA0DD
    style Output fill:#90EE90
```

**Key Post-processing Steps:**
1. **Smoothing**: Gaussian filtering for noise reduction
2. **Slope Constraint**: Limit rate of change
3. **Physical Constraint**: tan(δ) range
4. **Consistency Correction**: Cox-Merz rule
5. **Uncertainty Propagation**: Maintain confidence intervals

---

## 📈 Data Flow Diagram

```mermaid
flowchart LR
    Raw[Raw Data<br/>1874 samples] --> |80/20 Split| Train[Training Set<br/>~1499 samples]
    Raw --> |80/20 Split| Val[Validation Set<br/>~375 samples]
    
    Train --> |Batch 64| TrainLoader[Train DataLoader]
    Val --> |Batch 64| ValLoader[Val DataLoader]
    
    TrainLoader --> Model[PINN Model]
    ValLoader --> Model
    
    Model --> Pred1[log_G' Standardized]
    Model --> Pred2[log_G'' Standardized]
    
    Pred1 --> |Inverse Transform| G1_log[log_G']
    Pred2 --> |Inverse Transform| G2_log[log_G'']
    
    G1_log --> |10^x| G1[G' Pa]
    G2_log --> |10^x| G2[G'' Pa]
    
    G1 --> |Post-processing| G1_final[G' Final]
    G2 --> |Post-processing| G2_final[G'' Final]
    
    style Raw fill:#FFB6C1
    style Model fill:#DDA0DD
    style G1_final fill:#90EE90
    style G2_final fill:#90EE90
```

---

## 🎯 Core Innovations

```mermaid
mindmap
    root((PINN<br/>Core Innovations))
        Physics Constraints
            Cox-Merz Rule
            tan_δ_ Range
            Slope Constraint
            Smoothness
            Thermodynamic Constraint
        Feature Engineering
            WLF Time-Temperature
            Molecular Weight Features
            Geometric Features
            Coupled Features
        Uncertainty Quantification
            MC Dropout
            200 Samples
            Confidence Intervals
        Post-processing
            Physical Consistency
            Gaussian Smoothing
            Iterative Correction
            Boundary Propagation
```

---

## 📊 Performance Monitoring Flow

```mermaid
flowchart TD
    A[Each Epoch] --> B[Record Training Loss]
    B --> B1[Total Loss]
    B --> B2[Data Loss]
    B --> B3[Physics Loss]
    B --> B4[Component Losses]
    
    A --> C[Record Validation Loss]
    C --> C1[Total Loss]
    C --> C2[Physics Loss]
    
    B4 --> D[Print Every 10 Epochs]
    C2 --> D
    
    D --> E[After Training Complete]
    E --> F[Plot Loss Curves]
    F --> F1[Total Loss Comparison]
    F --> F2[Physics Constraint Loss]
    F --> F3[Component Trends]
    
    C1 --> G{Compare Validation Loss}
    G -->|Improved| H[Save Best Model]
    G -->|Not Improved| I[Increment Patience]
    I --> J{patience >= 50?}
    J -->|Yes| K[Early Stopping]
    J -->|No| Continue[Continue Training]
    
    style A fill:#FFE4B5
    style K fill:#FFB6C1
    style H fill:#90EE90
```

---

## 🔍 Complete System Summary

| Module | Input | Output | Key Technologies |
|--------|-------|--------|------------------|
| **Data Processing** | PB_Data.csv | Standardized Feature Matrix | Feature Engineering, WLF, Standardization |
| **Model Architecture** | 12D Features | 2D Prediction (log_G', log_G'') | Deep Encoder, Dual-branch |
| **Training** | Train/Val Sets | Best Model Weights | Physics Constraints, Dynamic Weights, Early Stopping |
| **Prediction** | New Samples | G', G'', Uncertainty | MC Dropout, Post-processing |
| **Visualization** | Prediction Results | Various Plots | Scatter Plots, Cole-Cole Plots |

**System Advantages:**
✅ Combines physics knowledge with data-driven approach  
✅ Provides prediction uncertainty  
✅ Automatic physical consistency correction  
✅ Complete training monitoring  
✅ Robust optimization strategy  

**Actual Performance:**
- Training set: ~1499 samples
- Validation set: ~375 samples
- Training epochs: 1000 (complete)
- Convergence: Continuous improvement, no early stopping
