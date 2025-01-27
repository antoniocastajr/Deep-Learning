CS 577: Deep Learning

Taught by Yutong Wang, this course focuses on understanding the internals of deep learning frameworks such as PyTorch. It helped me understand concepts like backpropagation and revisit other important topics, such as loss functions, optimizers, and their applications. This course includes six assignments, one project, and two exams.

Main topics:
-Regression: from linear regression to kernel methods to neural networks
-Classification: loss functions, cross entropy, numerical stability
-Optimizers: momentum, adaptive methods
-Backpropagation
-Computer Vision 
-Natural language processing
-Hardware

Description of directories:

-Assignments: Contains all my solutions for each assignment.
	-First Assignment: Introduction to Latex and implementation of Perceptron Algorithm.
	-Second Assignment: Implementation of gradient descent and solving probability and regression problems.
	-Third Assignment: We worked with tensors (3 dimension data structures), implementing the forward and loss function, and derivatives of a given function.
	-Fourth Assignment: Backpropagation using the autograd library. This library provides derivatives of functions to complete the backpropagation process. Implemented gradient descent using this library while working with tensors.
	-Fifth Assignment: The homework focuses on implementing and analyzing a simplified TransformerBlock, which includes two key components: 
		-SingleHeadAttention Layer: Performs sequence-to-sequence mapping using self-attention. 
		-MLP Layer: Applies a single hidden layer with a ReLU activation to the attention output.

-Exams: Constains problems and solutions that helped me prepare for both exams.

-Lectures: Slides and notebooks used in each lecture.

-Project: Our project focused on developing and testing a simplified version of BERT. We pretrained the model with tasks like Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). Finally, we fine-tuned the model for sentiment classification using the IMDB dataset of movie reviews, achieving 81.91% accuracy.

