/*
 * Author: Shuhao Lai
 * Date: 12/27/2018
 * DigitClassifier.h
 */

#ifndef DIGITCLASSIFIER_H_
#define DIGITCLASSIFIER_H_

#include <utility>
#include <string>
#include <math.h>
#include <vector>

class DigitClassifier
{
public:
	typedef std::vector<std::pair<int, std::vector<double>>>labeledImages;
	typedef std::vector<std::vector<double>> twoDArray;

	/*
	 * Each element in structure represents a layer in the neural network
	 * such that each value is the number of neurons in that layer.
	 */
	DigitClassifier(const std::vector<int> & structure)
	{
		this->structure = structure;
		fillSystemRandomly();
	}

	/*
	 * Used when weights and biases have been predetermined.
	 */
	DigitClassifier(const std::string & path)
	{
		readIn(path);
	}

	/*
	 * Determines how accurate the neural network is for test samples.
	 */
	void evaluate(std::string path);

	//Used for testing or actual classification. Image parameter should have same dimensions as images we trained on.
	int classify(std::vector<double> inputs);

	//Trains neural network
	void train(std::string path, int epoch, int miniBatchSize, double eta)
	{
		SGD(getImages(path), epoch, miniBatchSize, eta);
	}

	//In the return type, each pair contains a label and a vector of the pixels for the image.
	labeledImages getImages(const std::string & path);

	/*
	 * Randomly fills weights and biases in neural network.
	 */
	void fillSystemRandomly();

	//Shuffles the order of the labeledImages
	void shuffleImages(labeledImages & images);

	//Uses a built in shuffler for vectors.
	void shuffleImagesImproved(labeledImages & images);

	//Feeds inputs into one layer and returns z values. Layer parameter should account for input layer.
	std::vector<double> feedForwardOnce(const std::vector<double> & inputs, int layers);

	//Trains neural network using stochastic gradient descent.
	void SGD(labeledImages images, int epoch, int miniBatchSize, double eta);

	//Updates weights and biases once using a minibatch.
	void updateSystem(labeledImages mini, double eta);

	//Finds error in all layers.
	void backpropagate(int layer, const std::vector<double> & preError, const twoDArray & zVals, twoDArray & totalErrors);

	//Computes the error for the last layer of the neural network.
	std::vector<double> lastLayerError(std::vector<double> zVals, std::vector<int> y);

	//Prints out weights and biases to a text file.
	void toString(std::string path);

	//Sets weights and biases manually.
	void readIn(std::string path);

	//Extracts doubles from a string where the delimiters are spaces.
	std::vector<double> extractDoubles(std::string line);

	//Transposes a matrix
	twoDArray transpose(twoDArray twoD);

	//multiplies two matrices together and returns the resulting matrix.
	twoDArray multiplyMatrices(twoDArray a, twoDArray b);

	//Multiplies two vectors using hadamard product. The vectors must be row vectors with same and one dimension.
	std::vector<double> hadamard(std::vector<double> a, std::vector<double> b)
	{
		std::vector<double> product;
		for(int i = 0; i < (int)a.size(); i ++)
			product.push_back(a[i]*b[i]);
		return product;
	}

	//Returns vector of activations given a vector of z values.
	std::vector<double> activations(std::vector<double> zVals)
	{
		for(double & zVal : zVals )
			zVal = sigmoid(zVal);
		return zVals;
	}

	//Computes sigmoidPrime for each z value in zVals.
	std::vector<double> sigmoidPrimeVec(std::vector<double> zVals)
	{
		for(double & zVal : zVals )
			zVal = sigmoidPrime(zVal);
		return zVals;
	}

	double sigmoid(double z)
	{
		return (double) 1 / (1 + exp(-z));
	}

	double sigmoidPrime(double z)
	{
		return exp(z) / pow((exp(z) + 1), 2);
	}

private:

	/*
	 * Each element in structure represents a layer in the neural network
	 * such that each value is the number of neurons in that layer.
	 */
	std::vector<int> structure;

	/*a 2D vector storing the weights between two layers is stored inside a vector.
	 *The weights are stored in the conventional structure; weights[#][1][2] gets the
	 *weights from neuron 1 (in # layer) and neuron 2 (in #-1 layer).
	 */
	std::vector<twoDArray> weights;

	/*
	 * Two vector storing the biases of the system.
	 * To access a specific bias; [layer][neuron number]
	 */
	twoDArray biases;

};

#endif /* DIGITCLASSIFIER_H_ */
