# zi2zi (forked): 使用有條件式對抗網路來實作中國書法大師 （中文翻譯）

## 來至原始專案的主要更新項目

* 我們都知道原始專案上，GAN可以很好地產生**打印字體**。因此我們很好奇GAN可不可以在**手寫字體**上學習得很好。以及它是否可以通過微調學習我們的筆跡樣式。
* 使用灰階圖片可以很好的加快訓練速度
* 提供**訓練數據**和預處理腳本以進行準備
* **根據您自己的筆跡生成手寫漢字**！！！ 基於Tesseract和jTessBoxEditor的OCR工具包，可讓您拍攝手寫圖片並將其用作微調數據
* 腳本可微調筆跡，且少數幾張圖片即可掌握樣式

## 實驗結果

下面是經過9000步後，GAN生成的屏幕截圖。左側是基本情況，右側是給定字體的GAN生成的示例。您可以說條件GAN確實學習了，給定字體的一些樣式特徵。

![img](assets/sample1.png)
![img](assets/sample2.png)

模型開始訓練後，您可以在 “experiments/” 文件夾下查看範例和日誌。

這是張tensorboard的屏幕截圖，根據準確率對於給定的訓練數據，進行9000步是適合的停止點。（您可以添加更多手寫字體作為訓練數據，以使模型訓練時間更長，**但建議是：不要過度擬合**，否則微調將無法進行）。

![img](assets/tensorboard.png)

## 環境

* Python 3
* Tensorflow 1.8

## 使用

您可以按照以下步驟操作，也可以直接運行`run.sh`

```bash
##########################
## 預處理
##########################

# 繪製樣本字體並保存到paired_images，大約10-20分鐘
PYTHONPATH=. python font2img.py


##########################
## 訓練和推理
##########################

# 訓練模型
PYTHONPATH=. python train.py --experiment_dir=experiments \
                --experiment_id=0 \
                --batch_size=64 \
                --lr=0.001 \
                --epoch=40 \
                --sample_steps=50 \
                --schedule=20 \
                --L1_penalty=100 \
                --Lconst_penalty=15

# 推理
PYTHONPATH=. python infer.py --model_dir=experiments/checkpoint/experiment_0_batch_32 \
                --batch_size=32 \
                --source_obj=experiments/data/val.obj \
                --embedding_ids=0 \
                --save_dir=save_dir/
```

要了解如何準備**您的筆跡**，請查看[handwriting_preparation/README.md](/handwriting_preparation)，它詳細介紹瞭如何使用tesseract和jTessBoxEditor。

![img](assets/jTessBoxEditor.png)

```bash
##########################
## 微調
##########################

# 生成配對圖像以進行微調
PYTHONPATH=. python font2img_finetune.py


# 訓練/微調模型
PYTHONPATH=. python train.py --experiment_dir=experiments_finetune \
                --experiment_id=0 \
                --batch_size=16 \
                --lr=0.001 \
                --epoch=10 \
                --sample_steps=1 \
                --schedule=20 \
                --L1_penalty=100 \
                --Lconst_penalty=15 \
                --freeze_encoder_decoder=1 \
                --optimizer=sgd \
                --fine_tune=67 \
                --flip_labels=1

PYTHONPATH=. python infer.py --model_dir=experiments_finetune/checkpoint/experiment_0 \
                --batch_size=32 \
                --source_obj=experiments_finetune/data/val.obj \
                --embedding_id=67 \
                --save_dir=save_dir/

PYTHONPATH=. python infer_by_text.py --model_dir=experiments_finetune/checkpoint/experiment_0 \
                --batch_size=32 \
                --embedding_id=67 \
                --save_dir=save_dir/
```

## 相關項目

我的感覺是CycleGan會更好地執行，因為它不需要成對的數據，並且據說可以學習字體中更多的抽象結構特徵。
你可以參考 [Generating Handwritten Chinese Characters using CycleGAN](https://arxiv.org/pdf/1801.08624.pdf) 和如何實現它 [https://github.com/changebo/HCCG-CycleGAN](https://github.com/changebo/HCCG-CycleGAN)

---

## 以下是原始專案作者的自述文件

---

## 介紹

使用GAN學習東亞語言字體. zi2zi(字到字) 是最近流行的 [**pix2pix**](https://github.com/phillipi/pix2pix) 模型對漢字的應用和擴展。

細節可以在 [**部落格文章**](https://kaonashi-tyc.github.io/2017/04/06/zi2zi.html) 找到。

## 網路結構

### 原始模型

![alt network](assets/network.png)

網絡結構基於pix2pix，其中添加了類別嵌入以及分別來自 [AC-GAN](https://arxiv.org/abs/1610.09585) 和 [DTN](https://arxiv.org/abs/1611.02200) 的兩個其他損失，類別損失和恆定損失。

### 使用Label Shuffling更新模型

![alt network](assets/network_v2.png)

經過足夠的訓練後，**d_loss** 將下降到接近零，並且模型的性能穩定。**Label Shuffling**通過向模型提出新挑戰來緩解此問題。

具體來說，在給定的最小批量中，對於相同的源字符集，我們生成了兩組目標字符：
一組具有正確的嵌入標籤，另一組具有經過label shuffling。改組後的集合可能將沒有相應的目標圖像來計算 **L1\_Loss**，但可以用作所有其他損失的良好來源，從而迫使模型進一步超出了所提供示例的範圍。根據經驗，label shuffling可以改善模型在看不見的數據上的泛化能力，並提供更好的細節，並減少所需的字符數。

您可以在 **train.py**腳本中設置 **flip_labels = 1** 來啟用label shuffling。建議您在**d_loss**為零附近的平線之後啟用此功能，以進行進一步調整。

## 畫廊

### 與真相比較

![img](assets/compare3.png)

### 毛筆字體

![img](assets/cj_mix.png)

### 草書（SNS聽眾要求）

![img](assets/cursive.png)

### 宋體/明朝體

![img](assets/mingchao4.png)

### 韓文

![img](assets/kr_mix_v2.png)

### 不同粗細字體

![img](assets/transition.png)

### 動畫

![img](assets/poem.gif)
![img](assets/ko_wiki.gif)
![img](assets/reddit_bonus_humor_easter_egg.gif)

## 如何使用

### 步驟0

請隨意下載大量字體

### 需求

* Python 2.7
* CUDA
* cudnn
* Tensorflow >= 1.0.1
* Pillow(PIL)
* numpy >= 1.12.1
* scipy >= 0.18.1
* imageio

### 預處理

為了避免IO瓶頸，必須進行預處理，以將數據醃製為二進制數據並在訓練期間保留在內存中。

首先運行以下命令以獲取字體圖像：

```sh
python font2img.py --src_font=src.ttf
                   --dst_font=tgt.otf
                   --charset=CN
                   --sample_count=1000
                   --sample_dir=dir
                   --label=0
                   --filter=1
                   --shuffle=1
```

提供了四個默認字符集：CN，CN_T（繁體），JP，KR。您也可以將其指向一個單行文件，它將在其中生成字符圖像。注意，強烈建議您使用 **filter** 選項，它將對一些字符進行預採樣，並過濾所有具有相同哈希值的圖像，通常表明該字符丟失。 **label**，指示與該字體關聯的類別嵌入中的索引，默認為0。

獲取所有圖像後，運行 **package.py**將圖像及其對應的標籤醃製為二進制格式：

```sh
python package.py --dir=image_directories
                  --save_dir=binary_save_directory
                  --split_ratio=[0,1]
```

運行此命令後，您將在save_dir下找到兩個對象 **train.obj** 和 **val.obj** 分別用於訓練和驗證。

### 實驗版面

```sh
experiment/
└── data
    ├── train.obj
    └── val.obj
```

在項目根目錄下創建一個 **experiment** 目錄，並在其中創建一個數據目錄以放置兩個二進製文件。 假設目錄佈局可實現更好的數據隔離，尤其是在您運行多個實驗的情況下。

### 訓練

要開始訓練，請運行以下命令

```sh
python train.py --experiment_dir=experiment
                --experiment_id=0
                --batch_size=16
                --lr=0.001
                --epoch=40
                --sample_steps=50
                --schedule=20
                --L1_penalty=100
                --Lconst_penalty=15
```

**schedule**此處表示在幾個時期之間，學習率將下降一半。train命令將在 **experiment_dir** 下創建 **sample,logs,checkpoint**目錄。如果目錄不存在，您必須檢查和管理訓練的進度。

## 推理和內插

訓練完成後，運行以下命令來推斷測試數據：

```sh
python infer.py --model_dir=checkpoint_dir/
                --batch_size=16
                --source_obj=binary_obj_path
                --embedding_ids=label[s] of the font, separate by comma
                --save_dir=save_dir/
```

您也可以使用以下命令進行內插：

```sh
python infer.py --model_dir= checkpoint_dir/
                --batch_size=10
                --source_obj=obj_path
                --embedding_ids=label[s] of the font, separate by comma
                --save_dir=frames/
                --output_gif=gif_path
                --interpolate=1
                --steps=10
                --uroboros=1
```

它將遍歷embedding_ids中指定的所有字體對，並按指定的步驟插入數量。

### 預訓練模型

可以在[此處](https://drive.google.com/open?id=0Bz6mX0EGe2ZuNEFSNWpTQkxPM2c)下載已保存的模型，該模型使用27種字體進行了訓練，僅保存了生成器以減小模型尺寸。您可以在此預訓練模型中使用編碼器以加快訓練過程。

## 致謝

繼承或重新定義至以下專案：

* [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow) by [yenchenlin](https://github.com/yenchenlin)
* [Domain Transfer Network](https://github.com/yunjey/domain-transfer-network) by [yunjey](https://github.com/yunjey)
* [ac-gan](https://github.com/buriburisuri/ac-gan) by [buriburisuri](https://github.com/buriburisuri)
* [dc-gan](https://github.com/carpedm20/DCGAN-tensorflow) by [carpedm20](https://github.com/carpedm20)
* [origianl pix2pix torch code](https://github.com/phillipi/pix2pix) by [phillipi](https://github.com/phillipi)

## 授權

Apache 2.0
