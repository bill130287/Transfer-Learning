# Transfer-Learning
## Purpose:
Transfer Learning for NIST digits(0~9) to NIST letters(a~j)
用keras建立一CNN model，先去訓練NIST裡面有關手寫數字的dataset，從0-9每個數字都有3000張的圖片；然後利用這個model，去訓練另一NIST裡面，有關手寫英文字母的資料子集，從a~j每個字母只有400張的圖片，由於其資料集較小，因此透過Transfer Learning的方法，避免其over-fitting的情況產生。
最後再利用小畫家自己手寫a~j的字母各一張，去讓Transfer Learning後的model分類我寫的分別為何種字母。

## Define
   * train.py: main function for training and transfer learning
   * predict.py: main function for prediction
   * 可以自己用小畫家寫字母(128 * 128的尺寸)，然後再餵進model測試
   * train_model.h5: The CNN model for training digits.
   * transfer_train_model.h5: The CNN model for letters
   * summary_report.docx: About accuracy and model summary
