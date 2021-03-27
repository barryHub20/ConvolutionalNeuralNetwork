#include "MLPNeuron.h"
#include "Utilities.h"

MLPNeuron::MLPNeuron()
{
	reset();
}
MLPNeuron::~MLPNeuron() {}

void MLPNeuron::initRandomize(int layer, int index, int totalWeights)
{
	reset();
	this->layer = layer;
	this->index = index;

	//bias
	bias = (double)(rand() % 100 + 0) / 100.0;

	// weights
	weights.clear();
	weightsGradient.clear();
	weights.resize(totalWeights);
	weightsGradient.resize(totalWeights);

	double divider = 0.0;
	divider = 10000000.0;	// found to be NEW optimized value

	// for each weight
	for (int i = 0; i < totalWeights; ++i)
	{
		// random value with a multiplier based on layer
		weights[i] = (double)(rand() % 100 + 0) / divider;

		if (rand() % 2 == 0)
		{
			weights[i] = -weights[i];
		}
	}
}

void MLPNeuron::initAsPixel(int layer, int index, double pixel)
{
	this->layer = layer;
	this->index = index;
	this->a = pixel;
}

// reset all except vectors
void MLPNeuron::reset()
{
	layer = index = 0;
	localGradient = bias = biasGradient = z = a = 0.0;
}

void MLPNeuron::forwardPass(const vector<MLPNeuron>& prevLayer)
{
	if (prevLayer.size() != weights.size()) {
		cout << "prevLayer and weights size mismatch!" << endl;
		return;
	}

	// Z = sum(Wi * Ai) + B
	z = 0.0;
	double gg = 0.0;
	for (int i = 0; i < prevLayer.size(); ++i)
	{
		z += prevLayer[i].a * weights[i];
		gg += prevLayer[i].a;
	}
	z += bias;
	a = ReLU_Function(z);

	/*if (layer == 1 && index == 5)
	{
		cout << a << endl;
	}*/
}

void MLPNeuron::forwardPassDropout()
{
	z = a = 0.0;
}

void MLPNeuron::backwardPassOutputLayer(const vector<MLPNeuron>& prevLayer, double cost_m)
{
	if (prevLayer.size() != weights.size()) {
		cout << "prevLayer and weights size mismatch!" << endl;
		return;
	}

	// weights update, following Google docs
	double eq1 = 2 * (a - cost_m);
	double eq2 = ReLU_Derivative(a);
	
	for (int i = 0; i < prevLayer.size(); ++i)
	{
		double eq3 = prevLayer[i].a;
		weightsGradient[i] = eq1 * eq2 * eq3;
	}

	// bias update
	biasGradient = eq1 * eq2;

	// local gradient
	localGradient = eq1 * eq2;
}

void MLPNeuron::backwardPass(const vector<MLPNeuron>& prevLayer, const vector<MLPNeuron>& nextLayer)
{
	// weights update, following Google docs
	double eq1 = 0.0;
	for (int i = 0; i < nextLayer.size(); ++i)
	{
		eq1 += nextLayer[i].localGradient * nextLayer[i].weights[index];
	}
	double eq2 = ReLU_Derivative(a);
	
	for (int i = 0; i < prevLayer.size(); ++i)
	{
		double eq3 = prevLayer[i].a;
		weightsGradient[i] = eq1 * eq2 * eq3;
	}

	// bias update
	biasGradient = eq1 * eq2;

	// local gradient
	localGradient = eq1 * eq2;

	/*if (layer == 1 && index == 5)
	{
		cout << eq2 << endl;
	}*/
}

void MLPNeuron::backwardPassDropout()
{
	biasGradient = localGradient = 0.0;
	for (int i = 0; i < weightsGradient.size(); ++i)
	{
		weightsGradient[i] = 0.0;
	}
}

void MLPNeuron::weightsUpdate()
{
	for (int i = 0; i < weights.size(); ++i)
	{
		weights[i] = weights[i] - weightsGradient[i] * learningRate;
	}
	/*if (layer == 1 && index == 5)
	{
		cout << weightsGradient[2] << endl;
	}*/

	bias -= biasGradient * learningRate;
}