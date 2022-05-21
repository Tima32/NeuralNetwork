#pragma once
#include <vector>

typedef double (ActiveF)(double);

class Net
{
public:
	class Neyron
	{
	public:
		static double Sigmoid(double d);
		static double SigmoidDX(double d);

		static double Threshold(double d);
		static double ThresholdDX(double d);

		static double Tanh(double d);
		static double TanhDX(double d);

		double output;  //выход нейрона
		std::vector<double> weights; //веса

		ActiveF* active{ &Sigmoid };
	};

public:
	Net(const std::vector<size_t>& size);

	const std::vector<Neyron>& predict();
	void train(double learning_rate, size_t epochs, double max_mse, const std::vector<std::vector<double>>& in, const std::vector<std::vector<double>>& out);
	void train(double learning_rate, const std::vector<double>& out);
	//private:
	std::vector<std::vector<Neyron>> net; //[layer][neuron]
};