#pragma once
#include "Utilities.h"
#include "Filter.h"

struct PoolNeuron
{
	double input;
	double inputActivated;
	int xIndex;
	int yIndex;

	PoolNeuron() {
		input = inputActivated = 0.0;
		xIndex = yIndex = -1;
	}
};

class CNNFilter : public Filter
{
public:
	double** weights;
	double** weightsGradient;
	double bias, biasGradient;
	int weightsSize;	// width and height
	bool padding;

public:
	// functions
	CNNFilter();
	~CNNFilter();

	// init
	void init(int weightsSize, int layer, int index);

	// backpropagation
	void backpropagation(PoolNeuron** prevPoolingLayer, int prevPoolingLayerWidth, int prevPoolingLayerHeight, double** convLayerDelta,
		int convLayerWidth, int convLayerHeight);
	double gradientForEachWeight(int m, int n, PoolNeuron** prevPoolingLayer, int prevPoolingLayerWidth, int prevPoolingLayerHeight,
		double** convLayerDelta, int convLayerWidth, int convLayerHeight);

	// weights update
	void weightUpdate();
};