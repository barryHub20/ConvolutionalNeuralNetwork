#include "CNN.h"

CNN::CNN()
{
	imageIndex = correctIndex = FCLayerSize = 0;
}

CNN::~CNN()
{
	for (int i = 0; i < filterList.size(); ++i)
	{
		for (int j = 0; j < filterList[i].size(); ++j)
		{
			delete filterList[i][j];
		}
	}
	for (int i = 0; i < convLayerList.size(); ++i)
	{
		for (int j = 0; j < convLayerList[i].size(); ++j)
		{
			delete convLayerList[i][j];
		}
	}
	for (int i = 0; i < poolingLayerList.size(); ++i)
	{
		for (int j = 0; j < poolingLayerList[i].size(); ++j)
		{
			delete poolingLayerList[i][j];
		}
	}
}

void CNN::init()
{
	CNNFilter* filterPtr = NULL;
	CNNConvLayer* convLayerPtr = NULL;
	CNNPoolingLayer* poolingLayerPtr = NULL;

	// impt for now: ensure filter is odd size and input layer is even size

	// layer 1-------------------------------------------------/
	filterList.push_back(vector<CNNFilter*>());
	convLayerList.push_back(vector<CNNConvLayer*>());
	poolingLayerList.push_back(vector<CNNPoolingLayer*>());
	convLayerPtr = new CNNConvLayer();
	convLayerPtr->init();
	convLayerList[0].push_back(convLayerPtr);
	poolingLayerPtr = new CNNPoolingLayer();
	poolingLayerPtr->init(convLayerPtr, false);
	poolingLayerList[0].push_back(poolingLayerPtr);

	// other layers
	addNewLayer(5, 11, true);
	addNewLayer(2, 7, true);

	// FC layer
	FCLayerSize = 0;
	for (int i = 0; i < convLayerList[convLayerList.size() - 1].size(); ++i)
	{
		FCLayerSize += convLayerList[convLayerList.size() - 1][i]->get1DSize();
	}
	FCLayerVector.resize(FCLayerSize, 0.0);

	// MLP
	vector<int> hiddenLayerSizes{ 100, 40 };
	// customDivider put to -1.0 to not use it
	myMlp.init(FCLayerSize, hiddenLayerSizes, 10, 0, -1.0);
}

void CNN::addNewLayer(int layersPerIndex, int filterSize, bool pooling)
{
	int lastIndex = convLayerList.size() - 1;

	// new layer vectors
	filterList.push_back(vector<CNNFilter*>());
	convLayerList.push_back(vector<CNNConvLayer*>());
	poolingLayerList.push_back(vector<CNNPoolingLayer*>());

	// each prev conv layer
	for (int i = 0; i < convLayerList[lastIndex].size(); ++i)
	{
		// layers per index
		for (int j = 0; j < layersPerIndex; ++j)
		{
			// filter
			CNNFilter* filterPtr = new CNNFilter();
			filterPtr->init(filterSize, lastIndex + 1, i);
			filterList[lastIndex + 1].push_back(filterPtr);
			// conv
			CNNConvLayer* convLayerPtr = new CNNConvLayer();
			convLayerPtr->init(convLayerList[lastIndex][i], filterPtr);
			convLayerList[lastIndex + 1].push_back(convLayerPtr);
			// pool
			CNNPoolingLayer* poolingLayerPtr = new CNNPoolingLayer();
			poolingLayerPtr->init(convLayerPtr, pooling);
			poolingLayerList[lastIndex + 1].push_back(poolingLayerPtr);
		}
	}
}

void CNN::loadImage(const vector<char>& contents, const vector<char>& labels, int imageIndex)
{
	this->imageIndex = imageIndex;
	this->correctIndex = readLabelFromMnist(labels, imageIndex);	// read the corr. digit for this mnist image

	// 1st layer load image
	convLayerList[0][0]->loadImage(contents, labels, imageIndex);

	// set correct image index on MLP
	myMlp.setCorrectIndex(this->imageIndex, this->correctIndex);
}

bool CNN::train(bool showCost, int iteration, int epoch, ofstream& outputStream)
{
	bool isCorrect = false;
	isCorrect = forwardPass(showCost, iteration, epoch, false, outputStream);
	backwardPass();
	weightsUpdate();
	return isCorrect;
}

bool CNN::forwardPass(bool showCost, int iteration, int epoch, bool testMode, ofstream& outputStream)
{
	// for all conv layers except last (last layer is fed into FC)
	for (int i = 0; i < convLayerList.size() - 1; ++i)
	{
		// for each conv layer of this layer
		for (int j = 0; j < convLayerList[i].size(); ++j)
		{
			CNNConvLayer* currConvLayer = convLayerList[i][j];
			
			// for each filter and resultant conv layer of the next layer
			for (int k = 0; k < filterList[i + 1].size(); ++k)
			{
				// if a filter points to currConvLayer, it will be convoluted and output to resultant conv layer (output)
				if (filterList[i + 1][k]->index == j)
				{
					currConvLayer->performConvForNextConvLayer(filterList[i + 1][k], convLayerList[i + 1][k]);
				}
			}
		}
	}

	// convert last conv layers to FC layer
	int counter = 0;
	for (int i = 0; i < convLayerList[convLayerList.size() - 1].size(); ++i)
	{
		CNNConvLayer* currConvLayer = convLayerList[convLayerList.size() - 1][i];
		for (int j = 0; j < currConvLayer->inputHeight; ++j)
		{
			for (int i = 0; i < currConvLayer->inputWidth; ++i)
			{
				FCLayerVector[counter++] = currConvLayer->inputActivated[j][i];
			}
		}
	}

	// pass FC layer to MLP
	myMlp.loadFCLayer(FCLayerVector, this->imageIndex, this->correctIndex);

	// MLP full training
	bool isCorrect = false;
	if (!testMode)
	{
		isCorrect = myMlp.train(showCost, iteration, epoch, outputStream);
	}
	else
	{
		myMlp.forwardPass();
	}
	return isCorrect;
}

void CNN::backwardPass()
{
	// MLP has already been trained and the gradients of the weights after FC layer have been derived
	
	// get the delta for FC layer -> last conv layer-------------------------------------------------//
	int totalLayers = convLayerList.size();
	int counter = 0;
	// for each last layer
	for (int i = 0; i < convLayerList[totalLayers - 1].size(); ++i)
	{
		convLayerList[totalLayers - 1][i]->deriveDeltaValuesFCLayer(counter, myMlp);
	}

	// for each filter of last layer-----------------------------------------------------------------//
	for (int i = 0; i < filterList[totalLayers - 1].size(); ++i)
	{
		CNNFilter* currFilter = filterList[totalLayers - 1][i];
		CNNConvLayer* prevConvLayer = convLayerList[totalLayers - 2][currFilter->index];	// pointed to
		CNNConvLayer* resultantConvLayer = convLayerList[totalLayers - 1][i];
		// backpropagation for last layer
		currFilter->backpropagation(prevConvLayer->inputActivated, prevConvLayer->inputWidth, prevConvLayer->inputHeight,
			resultantConvLayer->deltaValues, resultantConvLayer->inputWidth, resultantConvLayer->inputHeight);
	}

	// get the delta for remaining layers------------------------------------------------------------//
	for (int i = totalLayers - 2; i > 0; --i)
	{
		// for each conv layer (index)
		for (int j = 0; j < convLayerList[i].size(); ++j)
		{
			// derive delta
			CNNConvLayer* currConvLayer = convLayerList[i][j];
			currConvLayer->deriveDeltaValuesLayer(filterList[i + 1], convLayerList[i + 1], j);
		}
	}

	// get the gradients for the weights-------------------------------------------------------------//
	for (int i = totalLayers - 2; i > 0; --i)
	{
		// for each filter
		for (int j = 0; j < filterList[i].size(); ++j)
		{
			// derive gradient
			CNNFilter* currFilter = filterList[i][j];
			CNNConvLayer* prevConvLayer = convLayerList[i - 1][currFilter->index];
			CNNConvLayer* resultantConvLayer = convLayerList[i][j];
			currFilter->backpropagation(prevConvLayer->inputActivated, prevConvLayer->inputWidth, prevConvLayer->inputHeight,
				resultantConvLayer->deltaValues, resultantConvLayer->inputWidth, resultantConvLayer->inputHeight);
		}
	}
}

void CNN::weightsUpdate()
{
	// all filters update weights
	int totalLayers = convLayerList.size();
	for (int i = totalLayers - 1; i > 0; --i)
	{
		// for each filter
		for (int j = 0; j < filterList[i].size(); ++j)
		{
			// derive gradient
			CNNFilter* currFilter = filterList[i][j];
			currFilter->weightUpdate();
		}
	}
}

string CNN::logFileName()
{
	stringstream ss;
	ss << "CNN {";
	// for each layer
	for (int i = 1; i < filterList.size(); ++i)
	{
		// filters per layer, filter size, have pooling
		ss << "(" << filterList[i].size() << "," << filterList[i][0]->weightsSize << "," 
			<< (poolingLayerList[i][0]->pooling ? "yes" : "no") << ")";
		if (i != filterList.size() - 1)
		{
			ss << ",";
		}
	}
	ss << "} " << myMlp.logFileName() << ".txt";
	return ss.str();
}

void CNN::test(const vector<char>& contents, const vector<char>& labels, bool onlyShowAccuracyAtEnd, ofstream& outputStream)
{
	// test all 10k images
	int correctCounter = 0;
	for (int i = 0; i < 10000; ++i)
	{
		loadImage(contents, labels, i);
		forwardPass(false, i, 0, true, outputStream);	// testMode true to prevent MLP from training it's weights
		int lastLayerIndex = myMlp.layers.size() - 1;	// MLP
		int brightestNeuron = 0;
		double brightestNeuronVal = 0.0;
		for (int j = 0; j < myMlp.layers[lastLayerIndex].size(); ++j)
		{
			if (myMlp.layers[lastLayerIndex][j].a > brightestNeuronVal)
			{
				brightestNeuron = j;
				brightestNeuronVal = myMlp.layers[lastLayerIndex][j].a;
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
			cout << "CNN Accuracy%: " << ((double)correctCounter / (double)i) * 100.0 << endl;
		}
	}
	if (onlyShowAccuracyAtEnd)
	{
		cout << "CNN Accuracy%: " << ((double)correctCounter / 10000.0) * 100.0 << endl;
	}
}