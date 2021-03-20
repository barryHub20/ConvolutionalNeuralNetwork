#include "CNNPoolingLayer.h"

CNNPoolingLayer::CNNPoolingLayer()
{
	pooling = false;
	input = NULL;
	convLayer = NULL;
	inputWidth = inputHeight = 0;
}

CNNPoolingLayer::~CNNPoolingLayer()
{
	// delete dynamic arrays and pointers
	if (input && inputWidth != 0 || inputHeight != 0)
	{
		for (int y = 0; y < inputHeight; ++y)
		{
			delete[] input[y];
		}
		delete[] input;
		input = NULL;
		inputWidth = inputHeight = 0;
	}
	convLayer = NULL;
	pooling = false;
}

void CNNPoolingLayer::init(CNNConvLayer* convLayer, bool pooling)
{
	// init pooling based on conv layer size
	this->convLayer = convLayer;
	this->inputWidth = convLayer->inputWidth;
	this->inputHeight = convLayer->inputHeight;
	this->pooling = pooling;

	// if pooling (max), only 1/4 size of convLayer
	if (pooling)
	{
		this->inputWidth = convLayer->inputWidth / 2;
		this->inputHeight = convLayer->inputHeight / 2;
		input = new PoolNeuron * [this->inputHeight];
		for (int y = 0; y < this->inputHeight; ++y)
		{
			input[y] = new PoolNeuron[this->inputWidth];
		};
	}
	else
	{
		input = new PoolNeuron * [this->inputHeight];
		for (int y = 0; y < this->inputHeight; ++y)
		{
			input[y] = new PoolNeuron[this->inputWidth];

			// also set coord to same as conv layer
			for (int x = 0; x < this->inputWidth; ++x)
			{
				input[y][x].xIndex = x;
				input[y][x].yIndex = y;
			}
		};
	}
}

void CNNPoolingLayer::performPooling()
{
	if (pooling)
	{
		// max pooling
		for (int y = 0, poolY = 0; y < convLayer->inputHeight; poolY++, y += 2)
		{
			for (int x = 0, poolX = 0; x < convLayer->inputWidth; poolX++, x += 2)
			{
				double highestActivatedValue = -1000.0;
				int xIndex = -1, yIndex = -1;

				// for each 2x2 area of the conv, find the max
				for (int yi = y; yi < y + 2; ++yi)
				{
					for (int xi = x; xi < x + 2; ++xi)
					{
						if (convLayer->inputActivated[yi][xi] > highestActivatedValue)
						{
							highestActivatedValue = convLayer->inputActivated[yi][xi];
							yIndex = yi;
							xIndex = xi;
						}
					}
				}
				// assign highest activated neuron
				input[poolY][poolX].input = convLayer->input[yIndex][xIndex];
				input[poolY][poolX].inputActivated = convLayer->inputActivated[yIndex][xIndex];
				input[poolY][poolX].xIndex = xIndex;
				input[poolY][poolX].yIndex = yIndex;
			}
		};
	}
	else
	{
		for (int y = 0; y < convLayer->inputHeight; y++)
		{
			for (int x = 0; x < convLayer->inputWidth; x++)
			{
				input[y][x].input = convLayer->input[y][x];
				input[y][x].inputActivated = convLayer->inputActivated[y][x];
			}
		}
	}
}
