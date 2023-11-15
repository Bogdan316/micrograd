#include <iostream>
#include <cmath>
#include <functional>
#include <set>
#include <vector>
#include <memory>
#include <random>

#include "../module/module.hpp"

namespace micrograd
{
    template <class T>
    class Neuron : Module<T>
    {
        std::vector<std::shared_ptr<T>> _w;
        std::shared_ptr<T> _b;

    public:
        std::vector<std::shared_ptr<T>> get_parameters() override;
        T operator()(std::vector<T> &x) override;
        Neuron(int in);
    };

    template <class T>
    Neuron<T>::Neuron(int in)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-1, 1);
        for (size_t i = 0; i < in; i++)
        {
            _w.push_back(std::make_shared<T>(T(dis(gen))));
        }
        _b = std::make_shared<T>(T(0));
    }

    template <class T>
    std::vector<std::shared_ptr<T>> Neuron<T>::get_parameters()
    {
        std::vector<std::shared_ptr<T>> vec(_w.begin(), _w.end());
        vec.push_back(_b);

        return vec;
    }

    template <class T>
    T Neuron<T>::operator()(std::vector<T> &x)
    {
        T act = *_b;
        for (size_t i = 0; i < _w.size(); i++)
        {
            act = act + *_w[i] * x[i];
        }
        return act.relu();
    }
}