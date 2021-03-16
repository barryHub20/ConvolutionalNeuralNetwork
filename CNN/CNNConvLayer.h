#pragma once
#include "Layer.h"
#include "CNNFilter.h"
#include "MLP.h"

/************************************************************************************************************
Explanation for pooling and filter layer:
-index 0 of conv layer is the image itself
-index 0 of filterList will prelude the creation of it's respective L + 1 conv layer
-as such, index 0 of filterList leads to index 1 of convLayer
-index 0 of poolingList directly after index 0 of convLayerList

-> forward propagation
		index: 0  | 1  | 2
    convLayer:  0 |  1 |  2
  poolingList:   0|   1|   2
	   filter: 0  | 1  | 2

Note:
1) Make sure for each subsequent index, there can't be lesser filter or pooling layers than before
2) Filters per index are ordered in the order it is added to the list

Indexing system:
-usage of vector and indexes to keep track
-index of current layer is the order value of the conv in the prev layer that this current layer is a result of
*************************************************************************************************************/
class CNNConvLayer : public Layer
{
public:
	double** input;
	double** inputActivated;
	double** deltaValues;	// used to temp. store delta values
	int inputWidth, inputHeight;

public:
	// functions
	CNNConvLayer();
	~CNNConvLayer();

	// init
	void init();	// init as image
	void init(CNNConvLayer* prevConvLayer, CNNFilter* prevFilter);	// init conv layer with filter

	// each iteration
	void loadImage(const vector<char>& contents, const vector<char>& labels, int imageIndex);	// only for image layers

	// perform conv for next layer
	void performConvForNextConvLayer(CNNFilter* nextFilter, CNNConvLayer* nextConvLayer);

	// derive delta values
	void deriveDeltaValuesFCLayer(int& counter, MLP& myMlp);
	void deriveDeltaValuesLayer(vector<CNNFilter*>& nextLayerFilters, vector<CNNConvLayer*>& nextlayerConvLayers, int currIndex);	// normal layers
	void deltaValuesForEachFilter(CNNFilter* resultantFilter, CNNConvLayer* resultantConvLayer);
	double deltaForEachNeuron(int i, int j, CNNFilter* resultantFilter, CNNConvLayer* resultantConvLayer);

	// get
	int get1DSize();
};