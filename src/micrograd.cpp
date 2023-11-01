#include <iostream>
#include <cmath>
#include <functional>
#include <set>
#include <vector>

#include "micrograd.hpp"

std::vector<const Value *> Value::_sorted;
std::set<const Value *> Value::_visited;

std::ostream &operator<<(std::ostream &s, const Value &v)
{
    s << "Value(data=" << v.get_data() << ", op=" << v.get_op() << ", grad=" << v.get_grad() << ")";
    return s;
}

void Value::backward()
{
    *_grad = 1.0;
    _topo_sort(this);
    for (auto i = _sorted.rbegin(); i != _sorted.rend(); i++)
    {
        auto val = *i;
        val->_backward(val->_l_child, val->_r_child, val);
    }

    _visited.clear();
    _sorted.clear();
}

void Value::_topo_sort(const Value *root)
{
    if (root == nullptr)
    {
        return;
    }

    if (_visited.find(root) == _visited.end())
    {
        _visited.insert(root);
        _topo_sort(root->_l_child);
        _topo_sort(root->_r_child);
        _sorted.push_back(root);
    }
}

void Value::_to_grapviz(int &number, int parent_number) const
{
    parent_number = number;
    std::cout << "\tx" << number << " [label=<" << *this << ">];" << std::endl;

    if (_l_child)
    {
        number++;
        std::cout << "\tx" << parent_number << " -> "
                  << "x" << number << ";" << std::endl;
        _l_child->_to_grapviz(number, parent_number);
    }

    if (_r_child)
    {
        number++;
        std::cout << "\tx" << parent_number << " -> "
                  << "x" << number << ";" << std::endl;
        _r_child->_to_grapviz(number, parent_number);
    }
}

void Value::to_grapviz()
{
    std::cout << "digraph G {" << std::endl;
    int num = 0;
    _to_grapviz(num);
    std::cout << "}" << std::endl;
}

Value::Value(const Value &other)
{
    _data = new float(*other._data);

    _grad = new float(*other._grad);

    _backward = other._backward;

    _l_child = other._l_child;
    _r_child = other._r_child;

    _op = other._op;
}

Value &Value::operator=(const Value &other)
{
    _data = other._data;
    _grad = other._grad;
    _l_child = other._l_child;
    _r_child = other._r_child;
    _op = other._op;
    return *this;
}

Value Value::operator+(const Value &other)
{
    Value out = Value(*_data + *other._data, this, &other, "+");

    out._backward = [](const Value *thiz, const Value *other, const Value *out)
    {
        *thiz->_grad += *out->_grad;
        *other->_grad += *out->_grad;
    };

    return out;
}

Value Value::operator*(const Value &other)
{
    auto out = Value(*_data * *other._data, this, &other, "*");
    out._backward = [](const Value *thiz, const Value *other, const Value *out)
    {
        *thiz->_grad += *other->_data * *out->_grad;
        *other->_grad += *thiz->_data * *out->_grad;
    };

    return out;
}

Value Value::tanh()
{
    float x = *_data;
    float e = std::exp(2 * x);
    float t = (e - 1) / (e + 1);
    auto out = Value(t, this, nullptr, "tanh");

    out._backward = [t](const Value *thiz, const Value *other, const Value *out)
    {
        *thiz->_grad += (1 - std::pow(t, 2)) * *out->_grad;
    };

    return out;
}
