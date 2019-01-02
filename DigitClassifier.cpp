/*
 * Author: Shuhao Lai
 * Date: 12/27/2018
 * DigitClassifier.cpp
 */
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include "DigitClassifier.h"

using std::ifstream;
using std::string;
using std::stringstream;
using std::vector;
using std::cout;
using std::endl;
using std::pair;
using std::make_pair;

typedef vector<pair<int, vector<double>>> labeledImages;
typedef vector<vector<double>> twoDArray;

void DigitClassifier::evaluate(std::string path)
{
	labeledImages images = getImages(path);
	int good = 0, curTotal = 0;
	for (const pair<double, vector<double>> & img : images)
	{
		int result = classify(img.second);
		if (result == img.first)
			++good;
		++curTotal;
		//cout << "Accuracy: " << good << "/" << curTotal << endl;
	}
	cout << "Final Accuracy: " << ((double)good/curTotal * 100) << "%" << endl;

}

/*
 * Returns what the neural network classifies the image as.
 */
int DigitClassifier::classify(std::vector<double> inputs)
{
	if (structure[0] != (int) inputs.size())
		cout
				<< "Program will continue but training images' size and input image size are different."
				<< endl;
	for (int i = 1; i < (int) structure.size(); i++)
		inputs = activations(feedForwardOnce(inputs, i));
	int iOfHighestAct = 0;
	for (int i = 1; i < (int) inputs.size(); i++)
		if (inputs[i] > inputs[iOfHighestAct])
			iOfHighestAct = i;
	return iOfHighestAct;
}

labeledImages DigitClassifier::getImages(const std::string & path)
{
	labeledImages images;
	ifstream imagesPath(path);
	if (imagesPath.is_open())
	{
		string line;
		while (getline(imagesPath, line))
		{
			stringstream oneImage(line);
			int label;
			double onePixel;
			vector<double> pixels;
			char dummy;
			oneImage >> label;
			oneImage >> dummy; //To consume the commas.
			while (oneImage >> onePixel)
			{
				pixels.push_back(onePixel/255); //Dividing by 255 normalizes the data. CRUCIAL TO NORMALIZE TO PREVENT NAN.
				oneImage >> dummy; //To consume the commas.
			}
			pair<int, vector<double>> oneFormattedImage = make_pair(label,
					pixels);
			images.push_back(oneFormattedImage);
		}
		cout << "All images extracted" << endl;
		return images;
	}
	else
	{
		cout << "file could not be opened" << endl;
		return images;
	}
}

/*
 * Randomly fills weights and biases in neural network
 */
void DigitClassifier::fillSystemRandomly()
{
	/*Seeds the random number generator.
	 *Only needs to do it once because it starts the random number
	 *"booklet" at a random page.
	 */
	std::srand(std::time(0));
	for (int layer = 1; layer < (int) structure.size(); layer++)
	{
		vector<double> oneLayerOfBiases;
		twoDArray oneLayerOfWeights;
		int rows = structure[layer], cols = structure[layer - 1];
		for (int r = 0; r < rows; r++)
		{
			//Proper initial values are crucial. Improper values will cause NaN to occur.
			//Don't forget that weights and biases can be negative.
			/*
			 * Biases will be set to 0 while a couple initial weight ranges will be tested:
			 * sample 1: (-r, r) where r = 1/sqrt(d) and d represents the number of inputs.
			 * sample 2: (-r, r) where r = 4*sqrt(6/(fan-in + fan-out)) where fan-in is the number of inputs and
			 * 					 fan-out is the number of outputs.
			 */
			double ranBias = 0;
			oneLayerOfBiases.push_back(ranBias);
			vector<double> oneVecOfWeights;
			for (int c = 0; c < cols; c++)
			{
				double r = 4*sqrt(6.0/(structure[0]+structure.back()));
				//double r = 1/sqrt((double)structure[0]);
				double ranWeight = r*(((double)rand()/RAND_MAX)*2 - 1);
				oneVecOfWeights.push_back(ranWeight);
			}
			oneLayerOfWeights.push_back(oneVecOfWeights);
		}
		biases.push_back(oneLayerOfBiases);
		weights.push_back(oneLayerOfWeights);
	}
}

void DigitClassifier::shuffleImages(labeledImages & images)
{
	labeledImages shuffled;
	unsigned int size = images.size();
	std::srand(std::time(0));
	while (size != 0)
	{
		int ran = rand() % size;
		pair<double, vector<double>> oneImage = images[ran];
		shuffled.push_back(oneImage);
		images.erase(images.begin() + ran);
		--size;
	}
	images = shuffled;
}

void DigitClassifier::shuffleImagesImproved(labeledImages & images)
{
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine e(seed);
	std::shuffle(std::begin(images), std::end(images), e);
}

/*
 * return z values, not activation.
 */
vector<double> DigitClassifier::feedForwardOnce(const vector<double> & inputs,
		int layer)
{
	vector<double> zVals;
	for (int neuron = 0; neuron < structure[layer]; neuron++)
	{
		double z = 0; //double z; is wrong because z value is conserved.
		for (int w = 0; w < structure[layer - 1]; w++)
			z += weights[layer - 1][neuron][w] * inputs[w];
		z += biases[layer - 1][neuron];
		zVals.push_back(z);
	}
	return zVals;
}

void DigitClassifier::SGD(labeledImages images, int epoch, int miniBatchSize,
		double eta)
{
	for (int i = 0; i < epoch; i++)
	{
		cout << "Starting epoch: " << (i+1) << endl;
		shuffleImagesImproved(images);
		int startOfMini = 0;
		for (int i = 0; i < ceil((double) images.size() / miniBatchSize); i++)
		{
			labeledImages::iterator begin = images.begin() + startOfMini, end;
			if (startOfMini + miniBatchSize > (int) images.size())
				end = images.end();
			else
				end = images.begin() + startOfMini + miniBatchSize;
			startOfMini = startOfMini + miniBatchSize;
			//Note that the copy constructor is [begin, end).
			labeledImages mini(begin, end);
			updateSystem(mini, eta);
		}
	}
}

void DigitClassifier::updateSystem(labeledImages mini, double eta)
{
	vector<twoDArray> weightGradients;
	//Filling weightGradients with zeros.
	for (const twoDArray & layer : weights)
	{
		twoDArray oneLayer;
		for (const vector<double> & vec : layer)
		{
			vector<double> zeros(vec.size(), 0);
			oneLayer.push_back(zeros);
		}
		weightGradients.push_back(oneLayer);
	}

	twoDArray biasGradients;
	//Filling biasGradients with zeros.
	for (const vector<double> & vec : biases)
	{
		vector<double> zeros(vec.size(), 0);
		biasGradients.push_back(zeros);
	}

	for (pair<double, vector<double>> img : mini)
	{
		//Note that the vector at index 0 contains the z values for layer 1.
		twoDArray zVals;
		vector<double> inputs = img.second;
		for (int layer = 1; layer < (int) structure.size(); layer++)
		{
			vector<double> zValsForOneLayer = feedForwardOnce(inputs, layer);
			zVals.push_back(zValsForOneLayer);
			inputs = activations(zValsForOneLayer);
		}
		vector<int> label;
		for (int i = 0; i < structure[structure.size() - 1]; i++)
		{
			if (i == img.first)
				label.push_back(1);
			else
				label.push_back(0);
		}

		vector<double> ErrorForLastLayer = lastLayerError(zVals.back(), label);

		twoDArray totalErrors;
		//Note that totalErrors stores matrix of error for neural network in reverse layers.
		//totalError[0] return vector of errors from last layer.
		totalErrors.push_back(ErrorForLastLayer);

		//first parameter; minus 1 b/c last layer is accounted for, minus 1 b/c indexes start at 0,
		//and minus 1 b/c first layer in structure has no error and weights does not account for first layer.
		backpropagate(structure.size() - 3, ErrorForLastLayer, zVals, totalErrors);

		//adding to weightGradient
		for (int layer = 0; layer < (int) weights.size(); layer++)
			for (int neuron = 0; neuron < (int) weights[layer].size(); neuron++)
				for (int preNeuron = 0; preNeuron < (int) weights[layer][neuron].size();
						preNeuron++)
				{
					int totalErrorIndex = totalErrors.size() - 1 - layer;
					double preAct;
					if(layer == 0)
						preAct = img.second[preNeuron];
					else
						preAct = sigmoid(zVals[layer-1][preNeuron]);
					weightGradients[layer][neuron][preNeuron] += totalErrors[totalErrorIndex][neuron] * preAct;
				}

		//adding to biasGradients
		for (int layer = 0; layer < (int) biases.size(); layer++)
			for (int neuron = 0; neuron < (int) biases[layer].size(); neuron++)
			{
				int totalErrorIndex = totalErrors.size() - 1 - layer;
				biasGradients[layer][neuron] += totalErrors[totalErrorIndex][neuron];
			}

	}

	int size = mini.size();
	//Applying the change to weights
	for (int layer = 0; layer < (int) weights.size(); layer++)
		for (int neuron = 0; neuron < (int) weights[layer].size(); neuron++)
			for (int preNeuron = 0; preNeuron < (int) weights[layer][neuron].size();
					preNeuron++)
			{
				double normalizedWeight =
						weightGradients[layer][neuron][preNeuron] / size;
				weights[layer][neuron][preNeuron] -= eta * normalizedWeight;
			}

	//Applying the change to biases
	for (int layer = 0; layer < (int) biases.size(); layer++)
		for (int neuron = 0; neuron < (int) biases[layer].size(); neuron++)
		{
			double normalizedBias = biasGradients[layer][neuron] / size;
			biases[layer][neuron] -= eta * normalizedBias;

		}

	//cout << "weights and biases have been updated" << endl;
}

void DigitClassifier::backpropagate(int layer, const vector<double> & preError,
		const twoDArray & zVals, twoDArray & totalErrors)
{
	if (layer == -1)
		return;
	vector<double> sigmoidPrimeVector = sigmoidPrimeVec(zVals[layer]);
	twoDArray transposedWeights = transpose(weights[layer + 1]);
	twoDArray preErrorInTwoD;
	preErrorInTwoD.push_back(preError);
	preErrorInTwoD = transpose(preErrorInTwoD); //preError was a row vector before.
	vector<double> weightsTimesError = transpose(
			multiplyMatrices(transposedWeights, preErrorInTwoD))[0];
	vector<double> error = hadamard(weightsTimesError, sigmoidPrimeVector);
	totalErrors.push_back(error);
	backpropagate(layer - 1, error, zVals, totalErrors);
}

vector<double> DigitClassifier::lastLayerError(vector<double> zVals,
		vector<int> y)
{
	vector<double> acts = activations(zVals);
	vector<double> gradRespectToAct;
	for (int i = 0; i < (int) acts.size(); i++)
		gradRespectToAct.push_back(acts[i] - y[i]);
	vector<double> sigmoidPrimeVector = sigmoidPrimeVec(zVals);
	return hadamard(gradRespectToAct, sigmoidPrimeVector);
}

void DigitClassifier::toString(string path)
{
	std::ofstream out(path);
	out << structure.size() << endl;
	for (int i = 0; i < (int) structure.size(); i++)
		out << structure[i] << " ";
	out << endl;
	out << "Biases" << endl;
	out << biases.size() << endl;
	for (const vector<double> & vec : biases)
	{
		for (const double & bias : vec)
			out << bias << " ";
		out << endl;
	}

	out << "Weights" << endl;
	out << weights.size() << endl;
	for (vector<vector<double>> twoD : weights)
	{
		out << twoD.size() << endl;
		for (const vector<double> vec : twoD)
		{
			for (const double & weights : vec)
				out << weights << " ";
			out << endl;
		}
	}
}

void DigitClassifier::readIn(string path)
{
	ifstream in(path);
	if (in.is_open())
	{
		string line;
		int size;

		//Fills structure
		in >> size;
		double num;
		for (int i = 0; i < size; i++)
		{
			in >> num;
			structure.push_back((int) num);
		}
		getline(in, line); //consumes whitespace

		//Fills biases
		getline(in, line); //Reads header
		in >> size;	//Reads the number of rows there are for biases matrix.
		getline(in, line); //reads newline.
		for (int i = 0; i < size; i++)
		{
			getline(in, line);
			biases.push_back(extractDoubles(line));
		}

		//Fills weights
		getline(in, line);
		int numOfMatrices;
		in >> numOfMatrices;
		in >> size; //Number of rows in the upcoming matrix.
		getline(in, line); //consumes newline.
		for (int i = 0; i < numOfMatrices; i++)
		{
			twoDArray twoD;
			for (int ii = 0; ii < size; ii++)
			{
				getline(in, line);
				twoD.push_back(extractDoubles(line));
			}
			weights.push_back(twoD);
			in >> size;
			getline(in, line); //to consume newline.
		}
	}
	else
	{
		cout << "ReadIn File could not be opened" << endl;
	}
}

vector<double> DigitClassifier::extractDoubles(string line)
{
	vector<double> nums;
	stringstream ss(line);
	double oneDouble;
	while (ss >> oneDouble)
		nums.push_back(oneDouble);
	return nums;
}

twoDArray DigitClassifier::transpose(twoDArray twoD)
{
	twoDArray transposed;
	for (int c = 0; c < (int) twoD[0].size(); c++)
	{
		vector<double> oneR;
		for (int r = 0; r < (int) twoD.size(); r++)
			oneR.push_back(twoD[r][c]);
		transposed.push_back(oneR);
	}
	return transposed;
}

twoDArray DigitClassifier::multiplyMatrices(twoDArray a, twoDArray b)
{
	twoDArray mult;
	for (int i = 0; i < (int) a.size(); ++i)
	{
		vector<double> oneR;
		for (int j = 0; j < (int) b[0].size(); ++j)
		{
			double total = 0;
			for (int k = 0; k < (int) a[0].size(); ++k)
				total += a[i][k] * b[k][j];
			oneR.push_back(total);
		}
		mult.push_back(oneR);
	}
	return mult;
}

