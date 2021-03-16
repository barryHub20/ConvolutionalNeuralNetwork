#include "MLP.h"

MLP::MLP() { usingFCLayer = false; }
MLP::~MLP() {}

// set the layers to pre-defined sizes
// layerOffset: for when MLP is used in CNN
void MLP::init(int inputTotalPixels, const vector<int>& hiddenLayersSize, int totalOutputClasses, int layerOffset, double customDivider)
{
	// create the neurons
	layers.resize(hiddenLayersSize.size() + 2);
	layers[0].resize(inputTotalPixels);	// input layer resize
	for (int i = 1; i < layers.size() - 1; ++i)
	{
		// hidden layers size
		layers[i].resize(hiddenLayersSize[i - 1]);
	}
	layers[layers.size() - 1].resize(totalOutputClasses);	// output layer resize
	costLayer.resize(totalOutputClasses, 0);

	// init the neurons (except input layer)
	for (int i = 1; i < layers.size(); ++i) {
		for (int j = 0; j < layers[i].size(); ++j) {
			layers[i][j].initRandomize(i, j, layers[i - 1].size(), layerOffset, customDivider);
		}
	}
}

// load the image for forward propagation and/or training
void MLP::loadImage(const vector<char>& contents, const vector<char>& labels, int imageIndex)
{
	// load pixels to first layer of MLP
	vector<double> tempFirstLayer(784);
	LoadMnistImageToNeurons(contents, imageIndex, tempFirstLayer);
	for (int i = 0; i < layers[0].size(); ++i) {
		layers[0][i].initAsPixel(0, i, tempFirstLayer[i]);
	}

	// load the cost layer
	this->imageIndex = imageIndex;
	fill(costLayer.begin(), costLayer.end(), 0);
	this->correctIndex = readLabelFromMnist(labels, imageIndex);	// read the corr. digit for this mnist image
	costLayer[this->correctIndex] = 1.0;
}

// load the FC layer for forward propagation and/or training
void MLP::loadFCLayer(const vector<double>& FCLayer, int imageIndex, int correctImageIndex)
{
	// error check
	if (FCLayer.size() != layers[0].size())
	{
		cout << "MLP: FC layer size mismatch" << endl;
	}
	usingFCLayer = true;

	// load FC layer
	for (int i = 0; i < layers[0].size(); ++i) {
		layers[0][i].initAsPixel(0, i, FCLayer[i]);
	}

	// load the cost layer
	this->imageIndex = imageIndex;
	fill(costLayer.begin(), costLayer.end(), 0);
	this->correctIndex = correctImageIndex;
	costLayer[this->correctIndex] = 1.0;
}

void MLP::train(bool showCost, int iteration, int epoch)
{
	forwardPass();
	lossFunction(showCost, iteration, epoch);
	backwardPass();
	weightsUpdate();
}

void MLP::forwardPass()
{
	// forward pass (except input layer)
	for (int i = 1; i < layers.size(); ++i)
	{
		for (int j = 0; j < layers[i].size(); ++j)
		{
			layers[i][j].forwardPass(layers[i - 1]);
		}
	}
}

void MLP::lossFunction(bool showCost, int iteration, int epoch)
{
	// Etotal = sum1/2(target - output)^2
	double total = 0.0;
	int lastLayerIdx = layers.size() - 1;
	for (int i = 0; i < costLayer.size(); ++i)
	{
		total += (costLayer[i] - layers[lastLayerIdx][i].a) * (costLayer[i] - layers[lastLayerIdx][i].a) * 0.5;
	}
	if (showCost) {
		cout << "Iter: " << iteration << "  Epoch: " << epoch << "  Cost: " << total << endl;
	}
}


void MLP::backwardPass()
{
	// last layer
	int lastLayerIdx = layers.size() - 1;

	/*cout << "==============================================" << endl;
	for (int i = 0; i < layers[lastLayerIdx].size(); ++i)
	{
		cout << layers[lastLayerIdx][i].localGradient << endl;
	}
	cout << "==============================================" << endl;*/

	for (int i = 0; i < layers[lastLayerIdx].size(); ++i)
	{
		layers[lastLayerIdx][i].backwardPassOutputLayer(layers[lastLayerIdx - 1], costLayer[i]);
	}

	// all hidden layers
	for (int i = lastLayerIdx - 1; i > 0; --i)
	// for (int i = 1; i <= lastLayerIdx - 1; ++i)
	{
		for (int j = 0; j < layers[i].size(); ++j)
		{
			layers[i][j].backwardPass(layers[i - 1], layers[i + 1]);
		}
	}

	// first layer
	if (usingFCLayer)
	{
		vector<MLPNeuron> emptyVector;
		for (int j = 0; j < layers[0].size(); ++j)
		{
			layers[0][j].backwardPass(emptyVector, layers[1]);
		}
	}
}

void MLP::weightsUpdate()
{
	for (int i = 1; i < layers.size(); ++i)
	{
		for (int j = 0; j < layers[i].size(); ++j)
		{
			layers[i][j].weightsUpdate();
		}
	}
}

void MLP::saveToTextFile()
{

}

void MLP::test(const vector<char>& contents, const vector<char>& labels, bool onlyShowAccuracyAtEnd)
{
	// test all 10k images
	int correctCounter = 0;
	for (int i = 0; i < 10000; ++i)
	{
		loadImage(contents, labels, i);
		forwardPass();
		int lastLayerIndex = layers.size() - 1;
		int brightestNeuron = 0;
		double brightestNeuronVal = 0.0;
		for (int j = 0; j < layers[lastLayerIndex].size(); ++j)
		{
			if (layers[lastLayerIndex][j].a > brightestNeuronVal)
			{
				brightestNeuron = j;
				brightestNeuronVal = layers[lastLayerIndex][j].a;
			}
		}

		// does it match?
		// cout << brightestNeuron << " " << this->correctIndex << endl;
		if (brightestNeuron == this->correctIndex)
		{
			correctCounter++;
		}
		if (!onlyShowAccuracyAtEnd && i % 200 == 0 && i != 0 && correctCounter != 0)
		{
			cout << "MLP Accuracy%: " << ((double)correctCounter / (double)i) * 100.0 << endl;;
		}
	}
	if (onlyShowAccuracyAtEnd)
	{
		cout << "MLP Accuracy%: " << ((double)correctCounter / 10000.0) * 100.0 << endl;;
	}
}