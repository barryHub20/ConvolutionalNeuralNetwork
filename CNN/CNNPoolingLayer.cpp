#include "CNNPoolingLayer.h"

CNNPoolingLayer::CNNPoolingLayer()
{
	pooling = false;
	input = NULL;
	convInput = convInputActivated = NULL;
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
		convInput = convInputActivated = NULL;
		inputWidth = inputHeight = 0;
	}
	pooling = false;
}

void CNNPoolingLayer::init(int convLayerWidth, int convLayerHeight, double** convInput, double** convInputActivated, bool pooling)
{
	// init pooling based on conv layer size
	this->convInput = convInput;
	this->convInputActivated = convInputActivated;
	this->convLayerWidth = convLayerWidth;
	this->convLayerHeight = convLayerHeight;
	this->inputWidth = convLayerWidth;
	this->inputHeight = convLayerHeight;
	this->pooling = pooling;

	// if pooling (max), only 1/4 size of convLayer
	if (pooling)
	{
		this->inputWidth = convLayerWidth / 2;
		this->inputHeight = convLayerHeight / 2;
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
		int poolY = 0;
		int poolX = 0;
		for (int y = 0; y < convLayerHeight; y += 2)
		{
			for (int x = 0; x < convLayerWidth; x += 2)
			{
				double highestActivatedValue = -1000.0;
				int xIndex = -1, yIndex = -1;

				// for each 2x2 area of the conv, find the max
				for (int yi = y; yi < y + 2; ++yi)
				{
					for (int xi = x; xi < x + 2; ++xi)
					{
						if (convInputActivated[yi][xi] > highestActivatedValue)
						{
							highestActivatedValue = convInputActivated[yi][xi];
							yIndex = yi;
							xIndex = xi;
						}
					}
				}
				// assign highest activated neuron
				input[poolY][poolX].input = convInput[yIndex][xIndex];
				input[poolY][poolX].inputActivated = convInputActivated[yIndex][xIndex];
				input[poolY][poolX].xIndex = xIndex;
				input[poolY][poolX].yIndex = yIndex;
				poolX++;
			}
			poolX = 0.0;
			poolY++;
		}
	}
	else
	{
		for (int y = 0; y < convLayerHeight; y++)
		{
			for (int x = 0; x < convLayerWidth; x++)
			{
				input[y][x].input = convInput[y][x];
				input[y][x].inputActivated = convInputActivated[y][x];
			}
		}
	}
}

int CNNPoolingLayer::get1DSize()
{
	return inputWidth * inputHeight;
}
