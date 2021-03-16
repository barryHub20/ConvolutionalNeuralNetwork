#include "CNN.h"
#define TOTAL_EPOCH 15	// 50
#define TOTAL_ITERATIONS 60000	// 60000
using namespace std;

// digits
//string trainingImagesSet = "digits-mnist/train-images.idx3-ubyte";
//string trainingLabelsSet = "digits-mnist/train-labels.idx1-ubyte";
//string testImagesSet = "digits-mnist/t10k-images.idx3-ubyte";
//string testLabelsSet = "digits-mnist/t10k-labels.idx1-ubyte";
// fashion
string trainingImagesSet = "fashion-mnist/train-images-idx3-ubyte";
string trainingLabelsSet = "fashion-mnist/train-labels-idx1-ubyte";
string testImagesSet = "fashion-mnist/t10k-images-idx3-ubyte";
string testLabelsSet = "fashion-mnist/t10k-labels-idx1-ubyte";

/************************************************************************************************************
Main
*************************************************************************************************************/
void runMLP()
{
	// read pixels and labels
	// training set has 60k images of digits and labels
	// testing set has 10k images of digits and labels
	vector<char> contents;
	vector<char> labels;
	readMnistFile(trainingImagesSet, contents);
	readMnistFile(trainingLabelsSet, labels);

	// init
	MLP myMlp;
	vector<int> hiddenLayerSizes{ 100, 40 };
	// customDivider put to -1.0 to not use it
	myMlp.init(784, hiddenLayerSizes, 10, 0, -1.0);

	// train
	// train for 60k times
	for (int x = 0; x < TOTAL_EPOCH; ++x)
	{
		for (int z = 0; z < TOTAL_ITERATIONS; ++z)
		{
			// load image
			myMlp.loadImage(contents, labels, z);
			myMlp.train(z % 500 == 0, z, x);
		}
	}

	// save to text file


	// test
	vector<char> contents2;
	vector<char> labels2;
	readMnistFile(testImagesSet, contents2);
	readMnistFile(testLabelsSet, labels2);
	myMlp.test(contents2, labels2, true);
}

void runCNN()
{
	// read pixels and labels
	// training set has 60k images of digits and labels
	// testing set has 10k images of digits and labels
	vector<char> contents;
	vector<char> labels;
	readMnistFile(trainingImagesSet, contents);
	readMnistFile(trainingLabelsSet, labels);

	// init
	CNN myCnn;
	myCnn.init();

	// train
	// train for 60k times
	for (int x = 0; x < TOTAL_EPOCH; ++x)
	{
		for (int z = 0; z < TOTAL_ITERATIONS; ++z)
		{
			// load image
			myCnn.loadImage(contents, labels, z);
			myCnn.train(false, z, x);
		}
	}

	// test
	vector<char> contents2;
	vector<char> labels2;
	readMnistFile(testImagesSet, contents2);
	readMnistFile(testLabelsSet, labels2);
	myCnn.test(contents2, labels2, true);
}

void testImageAndLabel()
{
	vector<char> contents2;
	vector<char> labels2;
	readMnistFile(trainingImagesSet, contents2);
	readMnistFile(trainingLabelsSet, labels2);
	printImage(50123, contents2, labels2);
}

int main()
{
	// set rand seed
	srand(14);	// MLP will be 84% on this seed (14), so ideally CNN should too

	// runMLP();
	//for (int i = 0; i < 30; ++i)
	//{
	// runCNN();
	//}
	runCNN();
	// testImageAndLabel();
	/*int size = 27;
	int filterSize = 5;
	int padding = (filterSize - 1) / 2;
	int sizeWithoutPadding = size - (filterSize - 1);
	int sizeWithPadding = sizeWithoutPadding + padding * 2;

	cout << "Layer size: " << size << endl;
	cout << "filter size: " << filterSize << endl;
	cout << "padding size: " << padding << endl;
	cout << "sizeWithoutPadding: " << sizeWithoutPadding << endl;
	cout << "sizeWithPadding: " << sizeWithPadding << endl;*/
}