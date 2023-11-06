#include "src/micrograd.hpp"

using namespace micrograd;

void print_graph(Value val)
{
    val.backward();
    val.to_grapviz();
}

int main()
{
    Value x1 = 2.0;
    Value x2 = 0.0;

    Value w1 = -3.0;
    Value w2 = 1.0;

    Value b = 6.881373587095432;

    Value x1w1 = x1 * w1;
    Value x2w2 = x2 * w2;

    Value x1w1x2w2 = x1w1 + x2w2;
    Value n = x1w1x2w2 + b;
    Value o = n.tanh();

    Value oo;
    oo = o;
    print_graph(oo);

    Neuron nn = 2;
    for (auto p : nn.get_parameters())
    {
        std::cout << *p << std::endl;
    }
    std::vector<Value> v;
    v.push_back(Value(1));
    v.push_back(Value(1));
    std::cout << nn(v);

    return 0;
}