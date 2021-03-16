#include "CNNFilter.h"

CNNFilter::CNNFilter()
{
	weights = weightsGradient = NULL;
	bias = biasGradient = 0.0;
	weightsSize = 0;
	padding = true;
}

CNNFilter::~CNNFilter()
{
	if (weights && weightsGradient && weightsSize != 0)
	{
		for (int y = 0; y < weightsSize; ++y)
		{
			delete[] weights[y];
			delete[] weightsGradient[y];
		}
		delete[] weights;
		delete[] weightsGradient;
		weights = weightsGradient = NULL;
		weightsSize = 0;
	}
}

void CNNFilter::init(int weightsSize, int layer, int index)
{
	// layer and index
	if (layer == 0)
	{
		cout << "CNN Filter: filter can't be of layer 0" << endl;
		return;
	}
	if (weightsSize % 2 == 0)
	{
		cout << "CNN Filter: filter can't be of even size" << endl;
		return;
	}
	this->layer = layer;
	this->index = index;
	this->weightsSize = weightsSize;

	// create dynamic array
	weights = new double* [weightsSize];
	weightsGradient = new double* [weightsSize];
	for (int y = 0; y < weightsSize; ++y)
	{
		weights[y] = new double[weightsSize];
		weightsGradient[y] = new double[weightsSize];
	}

	//bias
	bias = (double)(rand() % 100 + 0) / 100.0;

	// randomize weights
	// For CNN 1 filter layer
	// double divider = 100000000.0 / (layer + 1);	// found to be optimized value
	// For CNN filter 2 filter layer
	// double divider = 10000000.0 / (layer + 1);	// found to be optimized value
	double divider = 100000000.0 / (layer + 1);	// found to be optimized value

	// for each weight
	for (int y = 0; y < weightsSize; ++y)
	{
		for (int x = 0; x < weightsSize; ++x)
		{
			// random value with a multiplier based on layer
			weights[y][x] = (double)(rand() % 100 + 0) / divider;

			if (rand() % 2 == 0)
			{
				weights[y][x] *= -1;
			}

			// init gradients to 0
			weightsGradient[y][x] = 0.0;
		}
	}
}

void CNNFilter::backpropagation(double** prevConvLayer, int prevConvLayerWidth, int prevConvLayerHeight, double** resultantConvLayerDelta,
	int resultantConvLayerWidth, int resultantConvLayerHeight)
{
	// reset
	for (int y = 0; y < weightsSize; ++y)
	{
		for (int x = 0; x < weightsSize; ++x)
		{
			weightsGradient[y][x] = 0.0;
		}
	}

	// derive gradient
	for (int n = 0; n < weightsSize; ++n)
	{
		for (int m = 0; m < weightsSize; ++m)
		{
			weightsGradient[n][m] = gradientForEachWeight(m, n, prevConvLayer, prevConvLayerWidth, prevConvLayerHeight,
				resultantConvLayerDelta, resultantConvLayerWidth, resultantConvLayerHeight);
		}
	}

	// bias gradient
	biasGradient = 0.0;
	for (int j = 0; j < resultantConvLayerHeight - weightsSize + 1; ++j)
	{
		for (int i = 0; i < resultantConvLayerWidth - weightsSize + 1; ++i)
		{
			biasGradient += resultantConvLayerDelta[j][i];
		}
	}
}

double CNNFilter::gradientForEachWeight(int m, int n, double** prevConvLayer, int prevConvLayerWidth, int prevConvLayerheight,
	double** resultantConvLayerDelta, int resultantConvLayerWidth, int resultantConvLayerHeight)
{
	int padding = (weightsSize - 1) / 2;
	double gradient = 0.0;
	// consult cnn backpropagation WITH PADDING.txt for more info
	// start from -padding, with padding applied the resultant layer's W/H will always be the same as this layer's
	for (int j = -padding; j < resultantConvLayerHeight - padding; ++j)
	{
		for (int i = -padding; i < resultantConvLayerWidth - padding; ++i)
		{
			double d1 = 0.0;
			double d2 = 0.0;

			// check if is within padding bounds
			if (j >= 0 && i >= 0)
			{
				d1 = resultantConvLayerDelta[j][i];
			}

			// check if is within padding bounds
			// min index will be 0 at min. If it's index is below 0 it's value will always be 0
			if (j + n >= 0 && i + m >= 0 && j + n < prevConvLayerheight && i + m < prevConvLayerWidth)
			{
				d2 = prevConvLayer[j + n][i + m];
			}
			gradient += d1 * d2;
			// gradient += abs(d1 * d2);
		}
	}
	return gradient;
}

void CNNFilter::weightUpdate()
{
	for (int y = 0; y < weightsSize; ++y)
	{
		for (int x = 0; x < weightsSize; ++x)
		{
			weights[y][x] = weights[y][x] - weightsGradient[y][x] * learningRate;
		}
	}

	bias -= biasGradient * learningRate;
}