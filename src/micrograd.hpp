#include <iostream>
#include <cmath>
#include <functional>
#include <set>
#include <vector>

class Value
{
    float *_data;
    float *_grad;
    Value *_l_child;
    Value *_r_child;

    std::string _op;
    std::function<void(Value *, Value *, Value *)> _backward = [](Value *l_child, Value *r_child, Value *out)
    { return; };

    void _to_grapviz(int &number, int parent_number = 0);

    static std::vector<Value *> _sorted;
    static std::set<Value *> _visited;
    static void _topo_sort(Value *root);

public:
    Value(float data, Value *l_child, Value *r_child, std::string op = "")
    {
        this->_data = new float(data);
        this->_grad = new float(0);
        this->_l_child = l_child;
        this->_r_child = r_child;
        this->_op = op;
    }
    Value(float data, std::string op = "") : Value(data, nullptr, nullptr, op){};

    ~Value()
    {
        delete _data;
        delete _grad;
    }
    Value(const Value &other);

    float get_data() const { return *_data; }
    float get_grad() const { return *_grad; }
    std::string get_op() const { return _op; }

    void backward();
    void to_grapviz();

    Value operator+(Value &other);
    Value operator*(Value &other);
    Value tanh();
};