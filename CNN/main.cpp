#include "CNN.h"
#define TOTAL_EPOCH 100	// 50
#define TOTAL_ITERATIONS 60000	// 60000
using namespace std;

string loggingFolderAddr = "../logging/";
// digits
string trainingImagesSet = "digits-mnist/train-images.idx3-ubyte";
string trainingLabelsSet = "digits-mnist/train-labels.idx1-ubyte";
string testImagesSet = "digits-mnist/t10k-images.idx3-ubyte";
string testLabelsSet = "digits-mnist/t10k-labels.idx1-ubyte";
// fashion
//string trainingImagesSet = "fashion-mnist/train-images-idx3-ubyte";
//string trainingLabelsSet = "fashion-mnist/train-labels-idx1-ubyte";
//string testImagesSet = "fashion-mnist/t10k-images-idx3-ubyte";
//string testLabelsSet = "fashion-mnist/t10k-labels-idx1-ubyte";

/************************************************************************************************************
Main
*************************************************************************************************************/
void runMLPtest(string testImagesSet, string testLabelsSet, vector<char>& contents, vector<char>& labels,
	ofstream& outputStream, MLP& myMlp)
{
	// test
	readMnistFile(testImagesSet, contents);
	readMnistFile(testLabelsSet, labels);
	myMlp.test(contents, labels, true, outputStream);
}

void runMLP()
{
	// read pixels and labels
	// training set has 60k images of digits and labels
	// testing set has 10k images of digits and labels
	vector<char> contents;
	vector<char> labels;
	readMnistFile(trainingImagesSet, contents);
	readMnistFile(trainingLabelsSet, labels);

	// test
	vector<char> contents2;
	vector<char> labels2;
	readMnistFile(testImagesSet, contents2);
	readMnistFile(testLabelsSet, labels2);

	// init
	MLP myMlp;
	vector<int> hiddenLayerSizes{ 100, 40 };
	vector<double> dropoutRates{ 0.0, 0.0 };
	// customDivider put to -1.0 to not use it
	myMlp.init(784, hiddenLayerSizes, dropoutRates, 10);

	// file i/o
	ofstream outputStream(loggingFolderAddr + myMlp.logFileName() + ".txt");

	if (outputStream.is_open())
	{
		// train
		// train for 60k times
		for (int x = 0; x < TOTAL_EPOCH; ++x)
		{
			int correctCounter = 0;
			for (int z = 0; z < TOTAL_ITERATIONS; ++z)
			{
				bool showMetrics = z % 500 == 0 && z != 0;
				bool doTest = z == 30000 || z == 59999;

				// load image
				myMlp.loadImage(contents, labels, z);
				if (myMlp.train(showMetrics, z, x, outputStream))
				{
					correctCounter++;
				}

				// accuracy
				if (showMetrics)
				{
					cout << "  MLP Accuracy%: " << setprecision(4) << ((double)correctCounter / z) * 100.0 << endl;
					outputStream << "  MLP Accuracy%: " << setprecision(4) << ((double)correctCounter / z) * 100.0 << endl;
				}

				// test
				if (doTest)
				{
					cout << "Iter: " << z << "  Epoch: " << x << " - ";
					outputStream << "Iter: " << z << "  Epoch: " << x << " - ";
					runMLPtest(testImagesSet, testLabelsSet, contents2, labels2, outputStream, myMlp);
				}
			}
		}
	}
	outputStream.close();
}

void runCNNtest(string testImagesSet, string testLabelsSet, vector<char>& contents, vector<char>& labels, 
	ofstream& outputStream, CNN& myCnn)
{
	// test
	readMnistFile(testImagesSet, contents);
	readMnistFile(testLabelsSet, labels);
	myCnn.test(contents, labels, true, outputStream);
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

	// test
	vector<char> contents2;
	vector<char> labels2;
	readMnistFile(testImagesSet, contents2);
	readMnistFile(testLabelsSet, labels2);

	// init
	CNN myCnn;
	myCnn.init();

	// file i/o
	ofstream outputStream(loggingFolderAddr + myCnn.logFileName() + ".txt");

	if (outputStream.is_open())
	{
		// train
		// train for 60k times
		for (int x = 0; x < TOTAL_EPOCH; ++x)
		{
			int correctCounter = 0;
			for (int z = 0; z < TOTAL_ITERATIONS; ++z)
			{
				bool showMetrics = z % 500 == 0 && z != 0;
				bool doTest = z == 30000 || z == 59999;

				// load image
				myCnn.loadImage(contents, labels, z);
				if (myCnn.train(showMetrics, z, x, outputStream))
				{
					correctCounter++;
				}

				// accuracy
				if (showMetrics)
				{
					cout << "  CNN Accuracy%: " << setprecision(4) << ((double)correctCounter / z) * 100.0 << endl;
					outputStream << "  CNN Accuracy%: " << setprecision(4) << ((double)correctCounter / z) * 100.0 << endl;
				}

				// test
				if (doTest)
				{
					cout << "Iter: " << z << "  Epoch: " << x << " - ";
					outputStream << "Iter: " << z << "  Epoch: " << x << " - ";
					runCNNtest(testImagesSet, testLabelsSet, contents2, labels2, outputStream, myCnn);
				}
			}
		}
	}

	outputStream.close();
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
	runCNN();

	/*vector<double> randValues(10, 0.0);
	randNormalDistribution(randValues, 0, 1);
	for (int i = 0; i < randValues.size(); ++i)
	{
		cout << randValues[i] << endl;
	}*/
}