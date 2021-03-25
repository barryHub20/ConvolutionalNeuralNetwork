#include "CNNConvLayer.h"

CNNConvLayer::CNNConvLayer()
{
	input = inputActivated = deltaValues = NULL;
	inputWidth = inputHeight = 0;
}

CNNConvLayer::~CNNConvLayer()
{
	// delete dynamic arrays and pointers
	if (input && inputActivated && deltaValues && inputWidth != 0 || inputHeight != 0)
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
void CNNConvLayer::init(CNNPoolingLayer* prevPoolingLayer, CNNFilter* prevFilter)
{
	// input check
	if (prevPoolingLayer->inputWidth % 2 != 0 || prevPoolingLayer->inputHeight % 2 != 0)
	{
		cout << "Conv layer must be of even size" << endl;
	}

	// params
	this->inputWidth = prevPoolingLayer->inputWidth - prevFilter->weightsSize + 1;
	this->inputHeight = prevPoolingLayer->inputHeight - prevFilter->weightsSize + 1;
	this->layer = prevFilter->layer;
	this->index = prevFilter->index;

	// if prev filter utilizes padding, resultant conv layer will be same in size as prev conv layer
	if (prevFilter->padding)
	{
		this->inputWidth = prevPoolingLayer->inputWidth;
		this->inputHeight = prevPoolingLayer->inputHeight;
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

void CNNConvLayer::performConv(CNNPoolingLayer* prevPoolingLayer, CNNFilter* filter)
{
	int padding = (filter->weightsSize - 1) / 2;

	if (filter->padding)
	{
		// for each stride (left -> right, top -> bottom)
		for (int j = -padding; j < prevPoolingLayer->inputHeight - padding; ++j)
		{
			for (int i = -padding; i < prevPoolingLayer->inputWidth - padding; ++i)
			{
				// convolution
				double Xvalue = 0.0;

				// for each weight (left -> right, top -> bottom)
				for (int n = 0; n < filter->weightsSize; ++n)	// y axis
				{
					for (int m = 0; m < filter->weightsSize; ++m)	// x axis
					{
						double weightValue = filter->weights[n][m];
						// i + m, j + n
						int xIndex = i + m;
						int yIndex = j + n;
						double inputValue = 0.0;
						if (yIndex >= 0 && yIndex < prevPoolingLayer->inputHeight && xIndex >= 0 && xIndex < prevPoolingLayer->inputWidth)
						{
							inputValue = prevPoolingLayer->input[yIndex][xIndex].inputActivated;
						}
						Xvalue += inputValue * weightValue;
					}
				}

				// save convolution value to resultant conv layer
				this->input[j + padding][i + padding] = Xvalue + filter->bias;
				this->inputActivated[j + padding][i + padding] = ReLU_Function(Xvalue + filter->bias);
			}
		}
	}
	else
	{
		cout << "TBC" << endl;
	}
}

void CNNConvLayer::deriveDeltaValuesFCLayer(int& counter, CNNPoolingLayer* poolingLayer, MLP& myMlp)
{
	// set all deltas to zeros
	for (int j = 0; j < inputHeight; ++j)
	{
		for (int i = 0; i < inputWidth; ++i)
		{
			deltaValues[j][i] = 0.0;
		}
	}

	// Y over X mapped to pooling indexes
	for (int j = 0; j < poolingLayer->inputHeight; ++j)
	{
		for (int i = 0; i < poolingLayer->inputWidth; ++i)
		{
			// get the respective gradient from the FC layer from MLP
			double delta = myMlp.layers[0][counter++].localGradient;

			// map to correct
			deltaValues[poolingLayer->input[j][i].yIndex][poolingLayer->input[j][i].xIndex] = delta;
			// cout << "FC delta: " << delta << endl;
		}
	}
}

void CNNConvLayer::deriveDeltaValuesLayer(CNNPoolingLayer* poolingLayer, vector<CNNFilter*>& nextLayerFilters, vector<CNNConvLayer*>& nextlayerConvLayers, int currIndex)
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
			deltaValuesForEachFilter(poolingLayer, nextLayerFilters[i], nextlayerConvLayers[i]);
		}
	}
}

void CNNConvLayer::deltaValuesForEachFilter(CNNPoolingLayer* poolingLayer, CNNFilter* filter, CNNConvLayer* convLayer)
{
	// for each neuron Xl we need to derive it's respective delta
	for (int j = 0; j < poolingLayer->inputHeight; ++j)
	{
		for (int i = 0; i < poolingLayer->inputWidth; ++i)
		{
			// REMOVE SOON
			// min index will be 0 at min. If it's index is below 0 it's value will always be 0
			/*int xMin = max(x - filter->weightsSize + 1, 0);
			int yMin = max(y - filter->weightsSize + 1, 0);
			int xMax = x;
			int yMax = y;*/

			// add to delta
			int yIndex = poolingLayer->input[j][i].yIndex;
			int xIndex = poolingLayer->input[j][i].xIndex;
			deltaValues[yIndex][xIndex] += deltaForEachNeuron(xIndex, yIndex, filter, convLayer);
		}
	}
}

double CNNConvLayer::deltaForEachNeuron(int i, int j, CNNFilter* filter, CNNConvLayer* convLayer)
{
	int k1 = filter->weightsSize;
	int k2 = filter->weightsSize;
	double delta = 0.0;

	// consult cnn backpropagation WITH PADDING.txt for more info
	int padding = (filter->weightsSize - 1) / 2;
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
			if (j - n + padding >= 0 && i - m + padding >= 0 && j - n + padding < convLayer->inputHeight 
				&& i - m + padding < convLayer->inputWidth)
			{
				d1 = convLayer->deltaValues[j - n + padding][i - m + padding];
			}

			double d2 = filter->weights[n][m];
			double d3 = ReLU_Derivative(input[j][i]);
			delta += d1 * d2 * d3;
		}
	}
	return delta;
}