#include "MLP.h"

MLP::MLP() { usingFCLayer = false; }
MLP::~MLP() {}

// set the layers to pre-defined sizes
void MLP::init(int inputTotalPixels, const vector<int>& hiddenLayersSize, const vector<double>& dropoutRates, int totalOutputClasses)
{
	// create the neurons
	layers.resize(hiddenLayersSize.size() + 2);
	this->dropoutRates.resize(dropoutRates.size() + 2);
	this->dropoutInputs.resize(dropoutRates.size() + 2);
	layers[0].resize(inputTotalPixels);	// input layer resize
	this->dropoutRates[0] = 0.0;

	for (int i = 1; i < layers.size() - 1; ++i)
	{
		// hidden layers size
		layers[i].resize(hiddenLayersSize[i - 1]);
		// dropout layer assign
		this->dropoutRates[i] = dropoutRates[i - 1];
		this->dropoutInputs[i].resize(hiddenLayersSize[i - 1], false);
	}
	layers[layers.size() - 1].resize(totalOutputClasses);	// output layer resize
	costLayer.resize(totalOutputClasses, 0);

	this->dropoutRates[this->dropoutRates.size() - 1] = 0.0;

	// init the neurons (except input layer)
	for (int i = 1; i < layers.size(); ++i) {
		for (int j = 0; j < layers[i].size(); ++j) {
			layers[i][j].initRandomize(i, j, layers[i - 1].size());
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

void MLP::setCorrectIndex(int imageIndex, int correctIndex)
{
	this->imageIndex = imageIndex;
	this->correctIndex = correctIndex;
}

bool MLP::train(bool showCost, int iteration, int epoch, ofstream& outputStream)
{
	bool isCorrect = false;
	forwardPass();
	isCorrect = lossFunction(showCost, iteration, epoch, outputStream);
	backwardPass();
	weightsUpdate();
	return isCorrect;
}

void MLP::forwardPass()
{
	// forward pass (except input layer)
	for (int i = 1; i < layers.size(); ++i)
	{
		for (int j = 0; j < layers[i].size(); ++j)
		{
			// is dropout?
			bool isDropout = false;
			if (i < layers.size() - 1)
			{
				dropoutInputs[i][j] = false;	// reset
				isDropout = dropoutInputs[i][j] = dropoutRates[i] * 100.0 > (rand() % 100);
			}

			// no dropout, continue convolution
			if (!isDropout)
			{
				layers[i][j].forwardPass(layers[i - 1]);
			}
			else
			{
				layers[i][j].forwardPassDropout();
			}
		}
	}
}

// return true if correct choice
bool MLP::lossFunction(bool showCost, int iteration, int epoch, ofstream& outputStream)
{
	// Etotal = sum1/2(target - output)^2
	double total = 0.0;
	int lastLayerIdx = layers.size() - 1;
	int highestActivation = -1000.0;
	int highestActivationIdx = -1;
	for (int i = 0; i < costLayer.size(); ++i)
	{
		total += (costLayer[i] - layers[lastLayerIdx][i].a) * (costLayer[i] - layers[lastLayerIdx][i].a) * 0.5;
		if (highestActivation < layers[lastLayerIdx][i].a)
		{
			highestActivation = layers[lastLayerIdx][i].a;
			highestActivationIdx = i;
		}
	}
	if (showCost) {
		cout << "Iter: " << setw(5) << left << iteration << "  Epoch: " << setw(2) << left << epoch << 
		 "  Cost: " << fixed << setprecision(10) << total;
	}
	return highestActivationIdx == correctIndex;
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
	{
		for (int j = 0; j < layers[i].size(); ++j)
		{
			if (!dropoutInputs[i][j])
			{
				layers[i][j].backwardPass(layers[i - 1], layers[i + 1]);
			}
			else
			{
				layers[i][j].backwardPassDropout();
			}
		}
	}

	// first layer
	if (usingFCLayer)
	{
		for (int j = 0; j < layers[0].size(); ++j)
		{
			layers[0][j].backwardPassFC(layers[1]);
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

string MLP::logFileName()
{
	stringstream ss;
	ss << "MLP {";
	for (int i = 1; i < layers.size() - 1; ++i)
	{
		ss << layers[i].size();
		if (i != layers.size() - 2)
		{
			ss << ",";
		}
	}
	ss << "}";

	return ss.str();
}

void MLP::test(const vector<char>& contents, const vector<char>& labels, bool onlyShowAccuracyAtEnd, ofstream& outputStream)
{
	// test all 10k images
	int correctCounter = 0;
	for (int i = 0; i < 10000; ++i)
	{
		loadImage(contents, labels, i);
		forwardPass();
		int lastLayerIndex = layers.size() - 1;
		// get the brightest neuron in last/output layer
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

		// does it match? If yes add to correctCounter
		if (brightestNeuron == this->correctIndex)
		{
			correctCounter++;
		}
		if (!onlyShowAccuracyAtEnd && i % 200 == 0 && i != 0 && correctCounter != 0)
		{
			cout << "Test dataset 10k: MLP Accuracy%: " << ((double)correctCounter / (double)i) * 100.0 << endl;;
		}
	}
	// only show accuracy one shot at end
	if (onlyShowAccuracyAtEnd)
	{
		cout << "Test dataset 10k: MLP Accuracy%: " << ((double)correctCounter / 10000.0) * 100.0 << endl;
		outputStream << "Test dataset 10k: MLP Accuracy%: " << ((double)correctCounter / 10000.0) * 100.0 << endl;
	}
}