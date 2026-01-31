Derivation of Weight Update Formulas for a Neural Network
Cross-Entropy Loss with Softmax Output (Multi-Class Classification)
note(you need to unzip the Darknet rar)


1.	Introduction
Artificial Neural Networks (ANNs) are widely used for multi-class classification problems. Training such networks requires minimizing a loss function that measures the discrepancy between predicted
outputs and true labels. While the Mean Squared Error (MSE) loss can be used, it is not optimal for
classification tasks. Instead, the cross-entropy loss combined with the softmax activation function in the output layer is the standard and mathematically well-justified choice.
This report derives the weight update formulas (Δw) for both the output layer and hidden layers of a feedforward neural network trained using backpropagation, when softmax activation and cross-
entropy loss are used. The derivation closely mirrors the MSE-based approach but highlights the key simplification that arises in the softmax–cross-entropy combination.


2.	Network Architecture and Notation
We consider a standard feedforward neural network consisting of:  	An input layer
 	One or more hidden layers
 	An output layer with softmax activation
Notation
 	i: index of neurons in the previous layer
 	j: index of neurons in a hidden layer
 	k: index of neurons in the output layer
Forward Pass Variables
 	xi: input to the network
 	wij: weight from neuron i to neuron j  	wjk: weight from neuron j to neuron k
 	zk: pre-activation (net input) of output neuron k  	ak: output of neuron k after softmax
 	tk: target (true) label for class k


3.	Softmax Activation Function
 
The softmax function converts raw scores into probabilities:


a  = softmax(z ) =
 



ezk
 


Properties:
 	0 ≤ ak ≤ 1  	∑k ak = 1
 
k	k
m
 
ezm
 
This makes softmax ideal for multi-class classification.


4.	Cross-Entropy Loss Function
For a single training example, the cross-entropy loss is defined as:
L = − ∑ tk ln(ak)
k

 
where:
 	tk  	tk
 

= 1 for the correct class
= 0 for all other classes
 

 

5.	Backpropagation: Output Layer Derivation
5.1	Objective
We aim to compute the gradient:
 



using the chain rule.
 
∂L

 
∂wjk
 

 

 
5.2	Chain Rule Expansion





5.3	Individual Derivatives
 

∂L
∂wjk
 


= ∂L
∂ak
 


⋅ ∂ak
∂zk
 


⋅  ∂zk
∂wjk
 
1.	Loss derivative:



2.	Softmax + Cross-Entropy Simplification
A key result:
 

∂L
∂ak



∂L

 
∂zk
 

= − tk
ak





= ak − tk
 
This simplification occurs because the Jacobian of softmax cancels terms from the cross-entropy gradient.
3.	Net input derivative:

 
 ∂zk 
∂wjk
 
= aj
 

 

 
5.4	Final Gradient (Output Layer)
 

∂L
∂wjk
 


= (ak − tk)aj
 

 

5.5	Weight Update Rule (Output Layer)
Using gradient descent with learning rate η:



6.	Backpropagation: Hidden Layer Derivation
6.1	Objective
Compute the gradient for hidden layer weights:
∂L
∂wij


6.2	Chain Rule Expansion
 
∂L
∂wij
 
= ∑ ∂L
k  ∂zk
 
∂zk
⋅
∂aj
 
∂aj
⋅
∂zj
 
∂zj
⋅
∂wij
 

 

 
6.3	Individual Derivatives
 	Output error term:



 	Weighted contribution:
 


∂L
∂zk
 



= ak − tk
 




Activation derivative (e.g., sigmoid):



Net input derivative:
 
∂zk
∂aj


∂aj
∂zj
 

= wjk




= f ′(zj )
 
 ∂zj 
∂wij
 

= ai
 

 

 
6.4	Hidden Layer Error Term
Define the hidden neuron error:
 


δj = f ′(zj) ∑(ak − tk)wjk
k
 

 

 
6.5	Final Gradient (Hidden Layer)





6.6	Weight Update Rule (Hidden Layer)
 

∂L
∂wij
 


= δjai
 

 
 

 

7.	Comparison with MSE Loss

Aspect	MSE	Cross-Entropy + Softmax
Output error	(ak − tk)f ′(zk)	ak − tk
Gradient complexity	Higher	Simplified
Numerical stability	Lower	Higher
Classification performance	Suboptimal	Optimal
The elimination of f ′(zk) in the output layer is the key advantage.


8.	Conclusion
This report presented a complete derivation of the weight update formulas for neural networks
trained using cross-entropy loss with softmax activation. By applying the chain rule step-by-step, we demonstrated that:
 	The output layer gradient simplifies to ak − tk
 	The hidden layer update retains the familiar backpropagation structure
 	The softmax–cross-entropy combination is mathematically elegant and computationally efficient
These properties explain why this loss-activation pairing is the standard choice for modern multi-class neural network training.
















Traffic Classification and Activity Prediction Using Neural Networks
Part 2: Practical Model Improvement (Assignment 2: Traffic Type Prediction)
Note(python code covers both 2 and 3 assignments)
2.1 Baseline Analysis and Improvement Strategy
The initial baseline model (Code 1) achieved a limited overall accuracy of approximately 70%. Several weaknesses were identified: insufficient preprocessing, lack of feature scaling, no handling of missing or infinite values, and an architecture prone to overfitting due to the absence of regularization techniques.
The improvement strategy (Code 2) focused on:
•	Data Preprocessing: Removing identifiers (IP, ports, timestamps), handling NaN and infinite values, and normalizing feature scales with StandardScaler.
•	Class Handling: Applying compute_class_weight to address class imbalance in both traffic type (Label 1) and activity type (Label 2).
•	Network Architecture: Incorporating BatchNormalization and Dropout layers to improve training stability and prevent overfitting.
•	Evaluation and Reporting: Introducing confusion matrices and detailed classification reports to allow per-class performance analysis.
________________________________________
2.2 Model Architecture
The improved neural network used for traffic type prediction consists of three hidden layers with 256, 128, and 64 neurons, respectively. BatchNormalization and Dropout (0.3 and 0.2) were applied after the first two hidden layers to stabilize learning and reduce overfitting. The output layer utilized Softmax activation to classify the four traffic types: Non-Tor, NonVPN, Tor, and VPN.
________________________________________
2.3 Results and Analysis (Label 1)
The improved model achieved an overall accuracy of 93%. Detailed class-wise performance is shown below:
Class	Precision	Recall	F1-Score	Support
NON-TOR	0.99	0.99	0.99	22089
NONVPN	0.82	0.80	0.81	4773
TOR	0.39	0.94	0.55	278
VPN	0.84	0.81	0.82	4584
Overall metrics:
•	Accuracy: 0.93
•	Macro Avg F1-Score: 0.79
•	Weighted Avg F1-Score: 0.94
Analysis:
The model demonstrates excellent performance in identifying Non-Tor traffic. However, Tor traffic shows a low precision (0.39) despite high recall (0.94), indicating that many flows predicted as Tor are actually misclassified from other classes. This highlights an area for future refinement, particularly in detecting minority traffic types.
Confusion Matrix (Label 1)
[Image: confusion_matrix_label1]
________________________________________
Part 3: Activity Type Prediction (Assignment 3: Multi-Class Classification)
3.1 Model Configuration
The same preprocessing pipeline and network architecture were applied to predict specific activity types (Label 2). The output layer was adjusted for eight distinct activity classes: AUDIO-STREAMING, BROWSING, CHAT, EMAIL, FILE-TRANSFER, P2P, VIDEO-STREAMING, and VOIP.
________________________________________
3.2 Results and Analysis (Label 2)
The model achieved an overall accuracy of 76%. Detailed class-wise performance is shown below:
Class	Precision	Recall	F1-Score	Support
AUDIO-STREAMING	0.87	0.63	0.73	4270
BROWSING	0.84	0.85	0.84	9292
CHAT	0.72	0.38	0.50	2326
EMAIL	0.43	0.30	0.35	1229
FILE-TRANSFER	0.52	0.68	0.59	2237
P2P	0.95	0.91	0.93	9704
VIDEO-STREAMING	0.48	0.64	0.55	1953
VOIP	0.27	0.88	0.42	713
Overall metrics:
•	Accuracy: 0.76
•	Macro Avg F1-Score: 0.61
•	Weighted Avg F1-Score: 0.76
Analysis:
The model performs exceptionally well for P2P and BROWSING traffic. However, low recall for EMAIL (0.30) and VOIP (0.88 precision disparity) indicates frequent misclassification due to limited representation in the dataset. This suggests a need for class-weighting, oversampling, or synthetic data generation in future iterations.
Confusion Matrix (Label 2)
[Image: /confusion_matrix_label2]
________________________________________
Conclusion
The project successfully addressed all assignment requirements and demonstrated the impact of careful data preprocessing and neural network design. Key outcomes include:
•	Traffic Type (Label 1): Accuracy improved from ~70% (baseline) to 93%, with strong performance in majority classes but challenges in minority Tor traffic.
•	Activity Type (Label 2): Accuracy reached 76%, with high performance for P2P and BROWSING, but low performance for minority activities such as EMAIL and VOIP.
Key improvements over the baseline included: feature scaling, handling missing/infinite values, class weighting, and regularization through BatchNormalization and Dropout. Confusion matrices provide clear guidance for future refinement, particularly for low-recall classes.
Overall, the study demonstrates a practical and robust pipeline for multi-class network traffic and activity classification using neural networks.

