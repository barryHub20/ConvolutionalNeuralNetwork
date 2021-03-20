#pragma once
#include "Utilities.h"
#include "Filter.h"

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
	void backpropagation(double** prevConvLayer, int prevConvLayerWidth, int prevConvLayerheight, double** convLayerDelta,
		int convLayerWidth, int convLayerHeight);
	double gradientForEachWeight(int m, int n, double** prevConvLayer, int prevConvLayerWidth, int prevConvLayerHeight, 
		double** convLayerDelta, int convLayerWidth, int convLayerHeight);

	// weights update
	void weightUpdate();
};