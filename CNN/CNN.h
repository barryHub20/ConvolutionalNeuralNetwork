#pragma once
#include "CNNPoolingLayer.h"
#define CNN_FULLY_CONNECTED_SIZE 50000

/************************************************************************************************************
Training:
init(...);
loop:
	loadImage(...);
	train();
saveToTextFile();
*************************************************************************************************************/
class CNN
{
public:
	// data
	int imageIndex, correctIndex;
	int FCLayerSize;

	// layers
	vector< vector<CNNFilter*> > filterList;
	vector< vector<CNNConvLayer*> > convLayerList;
	vector< vector<CNNPoolingLayer*> > poolingLayerList;

	// MLP
	MLP myMlp;

	// misc
	vector<double> FCLayerVector;

public:
	// functions
	CNN();
	~CNN();

	// init: totalLayers includes pooling layers
	void init();
	void addNewLayer(int layersPerIndex, int filterSize, bool pooling);

	// load image
	void loadImage(const vector<char>& contents, const vector<char>& labels, int imageIndex);

	// train
	bool train(bool showCost, int iteration, int epoch, ofstream& outputStream);
	bool forwardPass(bool showCost, int iteration, int epoch, bool testMode, ofstream& outputStream);
	void backwardPass();
	void weightsUpdate();

	// save results
	string logFileName();

	// test
	void test(const vector<char>& contents, const vector<char>& labels, bool onlyShowAccuracyAtEnd, ofstream& outputStream);
};

