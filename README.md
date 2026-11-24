# Neural Network from Scratch (NumPy only)

A two-layer neural network written with only NumPy (forward pass, backpropagation, gradient descent) and tested on breast cancer diagnosis.

## STAR
- **Situation:** Deep learning libraries hide how a network really learns, so it is easy to use them without understanding them.
- **Task:** Build a neural network with no TensorFlow or PyTorch, prove the gradients are correct, and compare it with standard scikit-learn models.
- **Action:** Wrote sigmoid output, tanh hidden layer, binary cross-entropy loss and backpropagation by hand (30 inputs, 16 hidden units, learning rate 0.1, 1,000 full-batch epochs) on the Breast Cancer Wisconsin dataset (569 patients, 30 features, 80/20 stratified split, features standardised). Verified backpropagation with a numerical gradient check. Compared against Logistic Regression and scikit-learn MLP and tried hidden sizes 2 to 32.
- **Result:** The gradient check passed (max difference 2.9e-10). The network reached **95.6% test accuracy** (99.1% train) against 63.2% for the majority baseline, matching scikit-learn MLP. It missed only 1 of 42 malignant cases.

## Results (test set, 114 patients)
| Model | Test accuracy |
|---|---|
| Majority class baseline | 63.16% |
| My NumPy network | 95.61% |
| scikit-learn MLP (16 tanh) | 95.61% |
| Logistic Regression | **98.25%** |

Confusion matrix of my network (rows = real, columns = predicted):
| | pred malignant | pred benign |
|---|---|---|
| real malignant | 41 | 1 |
| real benign | 4 | 68 |

Malignant recall 0.98, malignant precision 0.91, benign precision 0.99.

## Key insights
- Gradient check passed, so the hand-written backpropagation is correct.
- Test loss is lowest around epoch 500 (about 0.083) and rises slightly after (0.085 at epoch 900) while train loss keeps falling, which is a small sign of overfitting.
- Hidden size did not matter much: 2, 4 and 16 units gave 95.61%, 8 and 32 gave 94.74%. Each difference is 1 patient out of 114.
- Plain Logistic Regression beat the neural network here, so on this small linear-ish dataset the extra complexity does not help.

## Limitations
- One 80/20 split with only 114 test patients, so 1 patient equals about 0.9% accuracy.
- Full-batch gradient descent only, no regularisation or early stopping.

## How to run
1. Open `neural_network_from_scratch.ipynb` in Google Colab.
2. Runtime > Run all (no download needed, data comes with scikit-learn).

## Files
- `neural_network_from_scratch.ipynb` - executed notebook with outputs
- `requirements.txt`

## Author
Akshat Kesharwani
- Portfolio: https://akshatkesharwani-info.github.io/
- GitHub: https://github.com/akshatkesharwani-info
