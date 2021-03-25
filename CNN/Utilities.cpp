#include "Utilities.h"

double ReLU_Function(double x)
{
	return max(x, 0.0);
}

double ReLU_Derivative(double reluVal)
{
	if (reluVal > 0.0)
	{
		return 1.0;
	}
	else
	{
		return 0.0;
	}
}

void randNormalDistribution(vector<double>& storeValues, int mean, int variance)
{
	default_random_engine generator;
	normal_distribution<double> distribution(mean, variance);

	for (int i = 0; i < storeValues.size(); ++i) {
		double number = distribution(generator);
		storeValues[i] = number;
	}
}

void readMnistFile(string fileName, vector<char>& contents)
{
	ifstream in;
	in.open(fileName, ios::in | ios::binary);

	if (in.is_open())
	{
		// get the starting position
		streampos start = in.tellg();

		// go to the end
		in.seekg(0, ios::end);

		// get the ending position
		streampos end = in.tellg();

		// go back to the start
		in.seekg(0, ios::beg);

		// create a vector to hold the data that
		// is resized to the total size of the file
		contents.clear();
		contents.resize(static_cast<size_t>(end - start));
		
		// read it in
		in.read(&contents[0], contents.size());
		in.close();
	}
	else
	{
		cout << "Error opening file" << endl;
	}
}

void printImage(int imageIdx, vector<char>& contents, vector<char>& labels)
{
	// print out handwritten digit
	int offset = 16;	// first 16 bits of file is buffer data
	for (int z = imageIdx; z < imageIdx + 1; ++z) {
		for (int i = 0; i < 28; ++i) {
			for (int x = 0; x < 28; ++x) {
				int idx = (z * 28 * 28) + (i * 28) + x + offset;
				// convert to signed int
				std::bitset< 8 > value = (std::bitset< 8 >)contents[idx];
				int value2 = static_cast<int>(value.to_ulong());	// the unsigned base 10 value (255 = max, 0 = min)

				if (value2 == 0) {
					cout << '*' << ' ';
				}
				else {
					cout << (char)254 << ' ';	// ascii for block
				}
			}
			cout << endl;
		}
	}

	// first 8 bits of file is buffer data
	int labelOffset = 8;

	// convert to signed int
	std::bitset< 8 > value = (std::bitset< 8 >)labels[labelOffset + imageIdx];
	int labelVal = static_cast<int>(value.to_ulong());	// the unsigned base 10 value (255 = max, 0 = min)
	cout << "Label: " << labelVal << endl;
}

void LoadMnistImageToNeurons(const vector<char>& contents, int imageIdx, vector<double>& neurons)
{
	// mnist file first 16 bits are buffer
	int start = 16 + imageIdx * 784;
	int counter = 0;
	for (int i = start; i < start + 784; ++i)
	{
		// convert to signed int
		std::bitset< 8 > value = (std::bitset< 8 >)contents[i];
		neurons[counter] = (double)(static_cast<int>(value.to_ulong())) / 255.0;	// the unsigned base 10 value (255 = max, 0 = min)
		counter++;
	}
}

void LoadMnistImageToNeurons(const vector<char>& contents, int imageIdx, double** neurons)
{
	// mnist file first 16 bits are buffer
	int start = 16 + imageIdx * 784;
	for (int i = start; i < start + 784; i += 28)
	{
		for (int x = 0; x < 28; ++x)
		{
			// convert to signed int
			std::bitset< 8 > value = (std::bitset< 8 >)contents[i + x];
			neurons[(i - start) / 28][x] = (double)(static_cast<int>(value.to_ulong())) / 255.0;	// the unsigned base 10 value (255 = max, 0 = min)
		}
	}
}

int readLabelFromMnist(const vector<char>& labels, int imageIdx)
{
	// mnist file first 8 bits are buffer
	std::bitset< 8 > value = (std::bitset< 8 >)labels[8 + imageIdx];
	return static_cast<int>(value.to_ulong());	// the unsigned base 10 value (255 = max, 0 = min)
}