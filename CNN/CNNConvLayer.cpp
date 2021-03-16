#include "CNNConvLayer.h"

CNNConvLayer::CNNConvLayer()
{
	input = inputActivated = deltaValues = NULL;
	inputWidth = inputHeight = 0;
}

CNNConvLayer::~CNNConvLayer()
{
	// delete dynamic arrays and pointers
	if (input && inputActivated && deltaValues && inputWidth != 0 && inputHeight != 0)
	{
		for (int y = 0; y < inputHeight; ++y)
		{
			delete[] input[y];
			delete[] inputActivated[y];
			delete[] deltaValues[y];
		}
		delete[] input;
		delete[] inputActivated;
		delete[] deltaValues;
		input = inputActivated = NULL;
		inputWidth = inputHeight = 0;
	}
}

// init as image
void CNNConvLayer::init()
{
	// fixed for MNNST data
	inputWidth = inputHeight = 28;

	// image input is always 1st layer and index 0
	index = layer = 0;

	// create the dynamic arrays
	input = new double* [inputHeight];
	inputActivated = new double* [inputHeight];
	deltaValues = new double* [inputHeight];
	for (int y = 0; y < inputHeight; ++y)
	{
		input[y] = new double[inputWidth];
		inputActivated[y] = new double[inputWidth];
		deltaValues[y] = new double[inputWidth];
		for (int x = 0; x < inputWidth; ++x)
		{
			input[y][x] = 0.0;
			inputActivated[y][x] = 0.0;
			deltaValues[y][x] = 0.0;
		}
	}
}

// init with filter
void CNNConvLayer::init(CNNConvLayer* prevConvLayer, CNNFilter* prevFilter)
{
	// params
	this->inputWidth = prevConvLayer->inputWidth - prevFilter->weightsSize + 1;
	this->inputHeight = prevConvLayer->inputWidth - prevFilter->weightsSize + 1;
	this->layer = prevFilter->layer;
	this->index = prevFilter->index;

	// if prev filter utilizes padding, resultant conv layer will be same in size as prev conv layer
	if (prevFilter->padding)
	{
		this->inputWidth = prevConvLayer->inputWidth;
		this->inputHeight = prevConvLayer->inputWidth;
	}

	// create the dynamic array
	input = new double* [inputHeight];
	inputActivated = new double* [inputHeight];
	deltaValues = new double* [inputHeight];
	for (int y = 0; y < inputHeight; ++y)
	{
		input[y] = new double[inputWidth];
		inputActivated[y] = new double[inputWidth];
		deltaValues[y] = new double[inputWidth];
		for (int x = 0; x < inputWidth; ++x)
		{
			input[y][x] = 0.0;
			inputActivated[y][x] = 0.0;
			deltaValues[y][x] = 0.0;
		}
	}
}

void CNNConvLayer::loadImage(const vector<char>& contents, const vector<char>& labels, int imageIndex)
{
	// load pixels to first layer of MLP
	LoadMnistImageToNeurons(contents, imageIndex, input);
	for (int y = 0; y < inputHeight; ++y)
	{
		for (int x = 0; x < inputWidth; ++x)
		{
			inputActivated[y][x] = input[y][x];
		}
	}
}

void CNNConvLayer::performConvForNextConvLayer(CNNFilter* nextFilter, CNNConvLayer* nextConvLayer)
{
	int padding = (nextFilter->weightsSize - 1) / 2;

	if (nextFilter->padding)
	{
		// for each stride (left -> right, top -> bottom)
		for (int j = -padding; j < this->inputHeight - padding; ++j)
		{
			for (int i = -padding; i < this->inputWidth - padding; ++i)
			{
				// convolution
				double Xvalue = 0.0;

				// for each weight (left -> right, top -> bottom)
				for (int n = 0; n < nextFilter->weightsSize; ++n)	// y axis
				{
					for (int m = 0; m < nextFilter->weightsSize; ++m)	// x axis
					{
						double weightValue = nextFilter->weights[n][m];
						// i + m, j + n
						int xIndex = i + m;
						int yIndex = j + n;
						double inputValue = 0.0;
						if (yIndex >= 0 && yIndex < this->inputHeight && xIndex >= 0 && xIndex < this->inputWidth)
						{
							inputValue = this->inputActivated[yIndex][xIndex];
						}
						Xvalue += inputValue * weightValue;
					}
				}

				// save convolution value to resultant conv layer
				nextConvLayer->input[j + padding][i + padding] = Xvalue + nextFilter->bias;
				nextConvLayer->inputActivated[j + padding][i + padding] = ReLU_Function(Xvalue + nextFilter->bias);
			}
		}
	}
	else
	{
		cout << "TBC" << endl;
	}
}

void CNNConvLayer::deriveDeltaValuesFCLayer(int& counter, MLP& myMlp)
{
	// Y over X
	for (int j = 0; j < inputHeight; ++j)
	{
		for (int i = 0; i < inputWidth; ++i)
		{
			// get the respective gradient from the FC layer from MLP
			double delta = myMlp.layers[0][counter++].localGradient;
			deltaValues[j][i] = delta;
			// cout << "FC delta: " << delta << endl;
		}
	}
}

void CNNConvLayer::deriveDeltaValuesLayer(vector<CNNFilter*>& nextLayerFilters, vector<CNNConvLayer*>& nextlayerConvLayers, int currIndex)
{
	// reset
	for (int j = 0; j < inputHeight; ++j)
	{
		for (int i = 0; i < inputWidth; ++i)
		{
			deltaValues[j][i] = 0.0;
		}
	}

	// for each filter & conv layer in next layer
	for (int i = 0; i < nextLayerFilters.size(); ++i)
	{
		// if is pointing to THIS conv layer
		if (currIndex == nextLayerFilters[i]->index)
		{
			// derive delta and add to delta map
			deltaValuesForEachFilter(nextLayerFilters[i], nextlayerConvLayers[i]);
		}
	}
}

void CNNConvLayer::deltaValuesForEachFilter(CNNFilter* resultantFilter, CNNConvLayer* resultantConvLayer)
{
	// for each neuron Xl we need to derive it's respective delta
	for (int j = 0; j < inputHeight; ++j)
	{
		for (int i = 0; i < inputWidth; ++i)
		{
			// REMOVE SOON
			// min index will be 0 at min. If it's index is below 0 it's value will always be 0
			/*int xMin = max(x - resultantFilter->weightsSize + 1, 0);
			int yMin = max(y - resultantFilter->weightsSize + 1, 0);
			int xMax = x;
			int yMax = y;*/

			// add to delta
			deltaValues[j][i] += deltaForEachNeuron(i, j, resultantFilter, resultantConvLayer);
		}
	}
}

double CNNConvLayer::deltaForEachNeuron(int i, int j, CNNFilter* resultantFilter, CNNConvLayer* resultantConvLayer)
{
	int k1 = resultantFilter->weightsSize;
	int k2 = resultantFilter->weightsSize;
	double delta = 0.0;

	// consult cnn backpropagation WITH PADDING.txt for more info
	int padding = (resultantFilter->weightsSize - 1) / 2;
	int qMinX = (i - k1 + 1) + padding;
	int qMaxX = (i) + padding;
	int qMinY = (j - k2 + 1) + padding;
	int qMaxY = (j) + padding;
	int rangeX = qMaxX - qMinX + 1;
	int rangeY = qMaxY - qMinY + 1;

	// iterate over m,n of Q range
	for (int n = 0; n < rangeY; ++n)
	{
		for (int m = 0; m < rangeX; ++m)
		{
			double d1 = 0.0;
			if (j - n + padding >= 0 && i - m + padding >= 0 && j - n + padding < resultantConvLayer->inputHeight 
				&& i - m + padding < resultantConvLayer->inputWidth)
			{
				d1 = resultantConvLayer->deltaValues[j - n + padding][i - m + padding];
			}
			// REMOVE SOON
			//// if exceed into padding
			//if (j - n >= 0 && i - m >= 0)
			//{
			//	d1 = resultantConvLayer->deltaValues[j - n][i - m];
			//}

			double d2 = resultantFilter->weights[n][m];
			double d3 = ReLU_Derivative(input[j][i]);
			delta += d1 * d2 * d3;
		}
	}
	return delta;
}

int CNNConvLayer::get1DSize()
{
	return inputWidth * inputHeight;
}