#include "micrograd.hpp"

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

    o.backward();
    o.to_grapviz();

    return 0;
}